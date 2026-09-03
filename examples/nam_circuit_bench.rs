//! Replay Nam-26 IBM-transpiled fixtures through PathSum (apply + reduce).
//!
//! This is the AGES-shaped engine cost: structured RZ/SX/CX Toffoli ladders
//! with a handful of live path variables, not the SX-saturated random windows
//! in `pathsum_benchmarking`.
//!
//! Usage:
//!   cargo run --release --example nam_circuit_bench -- /path/to/quasar_nam
//!
//! Prints per-circuit kgates/s, peak live path variables, and the share of
//! gates that see an H-free state (gauge is then a single early return).

use egglog::engine::engine_64::EvaluatedPathSum;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const REPS: usize = 80;
const WARMUP: usize = 8;

#[derive(Clone, Copy)]
enum Gate {
    X(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    H(usize),
    Sx(usize),
    Rz(usize, f64),
    Cx(usize, usize),
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn parse_angle(raw: &str) -> f64 {
    let s = raw.trim().replace(' ', "");
    let neg = s.starts_with('-');
    let s = s.trim_start_matches('+').trim_start_matches('-');
    let val = if s.eq_ignore_ascii_case("pi") {
        std::f64::consts::PI
    } else if let Some(rest) = s.strip_prefix("pi/") {
        std::f64::consts::PI / rest.parse::<f64>().expect("pi/denom")
    } else if let Some((num, den)) = s.split_once("*pi/") {
        num.parse::<f64>().expect("num") * std::f64::consts::PI / den.parse::<f64>().expect("den")
    } else if let Some(rest) = s.strip_suffix("*pi") {
        rest.parse::<f64>().unwrap_or(1.0) * std::f64::consts::PI
    } else {
        s.parse::<f64>().unwrap_or_else(|_| panic!("angle: {raw}"))
    };
    if neg { -val } else { val }
}

fn parse_qasm(path: &Path) -> (u32, Vec<Gate>) {
    let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut nq = 0u32;
    let mut gates = Vec::new();
    for line in src.lines() {
        let t = line.split("//").next().unwrap_or("").trim().trim_end_matches(';').trim();
        if t.is_empty() || t.starts_with("OPENQASM") || t.starts_with("include") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("qreg ") {
            if let Some(lb) = rest.find('[') {
                if let Some(rb) = rest.find(']') {
                    nq += rest[lb + 1..rb].parse::<u32>().unwrap();
                }
            }
            continue;
        }
        let lower = t.to_ascii_lowercase();
        let qubit = |s: &str| -> usize {
            let lb = s.find('[').unwrap();
            let rb = s.find(']').unwrap();
            s[lb + 1..rb].parse().unwrap()
        };
        if lower.starts_with("cx ") {
            let args = t[2..].trim();
            let (a, b) = args.split_once(',').unwrap();
            gates.push(Gate::Cx(qubit(a), qubit(b)));
        } else if lower.starts_with("rz(") {
            let close = t.find(')').unwrap();
            let angle = parse_angle(&t[3..close]);
            gates.push(Gate::Rz(qubit(&t[close + 1..]), angle));
        } else if lower.starts_with("sx ") {
            gates.push(Gate::Sx(qubit(&t[2..])));
        } else if lower.starts_with("x ") {
            gates.push(Gate::X(qubit(&t[1..])));
        } else if lower.starts_with("z ") {
            gates.push(Gate::Z(qubit(&t[1..])));
        } else if lower.starts_with("h ") {
            gates.push(Gate::H(qubit(&t[1..])));
        } else if lower.starts_with("sdg ") {
            gates.push(Gate::Sdg(qubit(&t[3..])));
        } else if lower.starts_with("tdg ") {
            gates.push(Gate::Tdg(qubit(&t[3..])));
        } else if lower.starts_with("s ") {
            gates.push(Gate::S(qubit(&t[1..])));
        } else if lower.starts_with("t ") {
            gates.push(Gate::T(qubit(&t[1..])));
        }
    }
    (nq, gates)
}

fn apply(state: &mut EvaluatedPathSum, g: Gate) {
    match g {
        Gate::X(q) => state.apply_x(q),
        Gate::Z(q) => state.apply_z(q),
        Gate::S(q) => state.apply_s(q),
        Gate::Sdg(q) => state.apply_sdg(q),
        Gate::T(q) => state.apply_t(q),
        Gate::Tdg(q) => state.apply_tdg(q),
        Gate::H(q) => state.apply_h(q),
        Gate::Sx(q) => state.apply_sx(q),
        Gate::Rz(q, th) => state.apply_rz(q, th),
        Gate::Cx(qc, qt) => state.apply_cx(qc, qt),
    }
    state.reduce();
}

fn replay(n: u32, gates: &[Gate]) -> (u32, u32, usize) {
    let mut state = EvaluatedPathSum::new_id(n);
    let mut peak_vars = 0u32;
    let mut hfree = 0u32;
    let mut peak_terms = 0usize;
    for &g in gates {
        if state.num_path_vars == 0 {
            hfree += 1;
        }
        apply(&mut state, g);
        peak_vars = peak_vars.max(state.num_path_vars);
        peak_terms = peak_terms.max(state.phase_poly.terms.len());
    }
    (peak_vars, hfree, peak_terms)
}

fn main() {
    let dir = PathBuf::from(
        env::args()
            .nth(1)
            .unwrap_or_else(|| "/agent/repos/ages/tests/fixtures/quasar_nam".into()),
    );
    let names = [
        "tof_3.qasm",
        "tof_4.qasm",
        "tof_5.qasm",
        "barenco_tof_3.qasm",
        "barenco_tof_4.qasm",
        "vbe_adder_3.qasm",
        "mod5_4.qasm",
        "csla_mux_3.qasm",
        "rc_adder_6.qasm",
        "mod_mult_55.qasm",
    ];
    println!(
        "nam_circuit_bench dir={} engine=engine_64 reps={REPS}",
        dir.display()
    );
    let mut total_gates = 0usize;
    let mut total_secs = 0.0f64;
    for name in names {
        let path = dir.join(name);
        if !path.exists() {
            println!("skip={name} (missing)");
            continue;
        }
        let (n, gates) = parse_qasm(&path);
        if gates.is_empty() {
            println!("skip={name} (no gates)");
            continue;
        }
        let (peak_vars, hfree, peak_terms) = replay(n, &gates);
        for _ in 0..WARMUP {
            let _ = replay(n, &gates);
        }
        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t0 = Instant::now();
            let _ = replay(n, &gates);
            times.push(t0.elapsed().as_secs_f64());
        }
        let med = median(times);
        let kgps = gates.len() as f64 / med / 1e3;
        total_gates += gates.len();
        total_secs += med;
        println!(
            "circuit={name} qubits={n} gates={} median_us={:.1} kgates_per_s={:.1} peak_path_vars={peak_vars} hfree_gate_frac={:.2} peak_phase_terms={peak_terms}",
            gates.len(),
            med * 1e6,
            kgps,
            hfree as f64 / gates.len() as f64,
        );
    }
    if total_secs > 0.0 {
        println!(
            "TOTAL gates={total_gates} secs={:.6} kgates_per_s={:.1}",
            total_secs,
            total_gates as f64 / total_secs / 1e3
        );
    }
}
