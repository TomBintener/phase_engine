//! Deterministic (seeded) micro-benchmarks for the PathSum phase engines.
//!
//! Two workload shapes, both over the AGES gate set {x, z, sx, rz, cx} on
//! 2-10 qubits, driven by seeded random windows of 100-200 gates:
//!
//! 1. `engine`: the pure engine loop (gate application + eager `reduce()` on
//!    an owned `EvaluatedPathSum`), measuring gate+reduce throughput and the
//!    peak `phase_poly` size.
//! 2. `ffi`: the FFI-shaped round trip the egglog primitives pay per gate:
//!    `BaseValues::unwrap` (an `InternTable::get_cloned` deep clone), the
//!    `pathsum.rs` gate logic (clone + gate + reduce), and `BaseValues::get`
//!    (intern: hash + Eq probe + clone-on-miss). Each window is replayed
//!    twice per repetition against the same intern table so both the
//!    miss-dominated (first pass) and hit-dominated (second pass) intern
//!    paths are measured; every repetition uses a fresh table.
//!
//! Run with: `cargo bench --bench pathsum_benchmarking`
//! All workloads are fixed-seed deterministic; only wall-clock timing varies.

use egglog::sort::BaseValues;
use std::time::Instant;

const SEED: u64 = 0x5EED_CAFE_F00D_0001;
const WINDOWS_PER_CONFIG: usize = 12;
const REPS: usize = 5;
/// Stop introducing fresh path variables (sx) beyond this many live path
/// vars; the gate is deterministically replaced by z. Keeps both engines
/// far away from their variable-capacity asserts.
const PATH_VAR_CAP: u32 = 24;

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const RZ_ANGLES: [f64; 10] = [
    std::f64::consts::FRAC_PI_4,
    std::f64::consts::FRAC_PI_2,
    -std::f64::consts::FRAC_PI_2,
    std::f64::consts::PI,
    3.0 * std::f64::consts::FRAC_PI_4,
    std::f64::consts::FRAC_PI_8,
    0.3,
    0.7,
    -1.1,
    2.5,
];

#[derive(Clone, Copy, Debug)]
enum Gate {
    X(usize),
    Z(usize),
    Sx(usize),
    Rz(usize, f64),
    Cx(usize, usize),
}

/// A seeded random window of 100-200 gates over {x, z, sx, rz, cx}.
fn gen_window(rng: &mut XorShift, n: usize) -> Vec<Gate> {
    let len = 100 + rng.below(101) as usize;
    (0..len)
        .map(|_| {
            let q = rng.below(n as u64) as usize;
            match rng.below(5) {
                0 => Gate::X(q),
                1 => Gate::Z(q),
                2 => Gate::Sx(q),
                3 => Gate::Rz(q, RZ_ANGLES[rng.below(RZ_ANGLES.len() as u64) as usize]),
                _ => {
                    let mut qc = rng.below(n as u64) as usize;
                    if qc == q {
                        qc = (qc + 1) % n;
                    }
                    Gate::Cx(qc, q)
                }
            }
        })
        .collect()
}

fn gen_windows(n: usize, engine_tag: u64) -> Vec<Vec<Gate>> {
    let mut rng = XorShift::new(SEED ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ engine_tag);
    (0..WINDOWS_PER_CONFIG).map(|_| gen_window(&mut rng, n)).collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

macro_rules! engine_bench {
    ($engine:ident, $tag:expr) => {{
        use egglog::engine::$engine::EvaluatedPathSum;
        let mut grand_total_gates = 0usize;
        let mut grand_total_secs = 0.0f64;
        for n in 2..=10usize {
            let windows = gen_windows(n, $tag);
            let total_gates: usize = windows.iter().map(|w| w.len()).sum();
            let mut peak_phase_terms = 0usize;
            let run_all = |peak: &mut usize| {
                for w in &windows {
                    let mut state = EvaluatedPathSum::new_zero_state(n as u32);
                    for g in w {
                        match *g {
                            Gate::X(q) => state.apply_x(q),
                            Gate::Z(q) => state.apply_z(q),
                            Gate::Sx(q) => {
                                if state.num_path_vars > PATH_VAR_CAP {
                                    state.apply_z(q)
                                } else {
                                    state.apply_sx(q)
                                }
                            }
                            Gate::Rz(q, theta) => state.apply_rz(q, theta),
                            Gate::Cx(qc, qt) => state.apply_cx(qc, qt),
                        }
                        state.reduce();
                        *peak = (*peak).max(state.phase_poly.terms.len());
                    }
                }
            };
            // Warmup
            run_all(&mut peak_phase_terms);
            let mut times = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let start = Instant::now();
                run_all(&mut peak_phase_terms);
                times.push(start.elapsed().as_secs_f64());
            }
            let med = median(times);
            grand_total_gates += total_gates;
            grand_total_secs += med;
            println!(
                "bench=engine engine={} qubits={} gates={} median_ms={:.3} kgates_per_s={:.1} peak_phase_terms={}",
                stringify!($engine),
                n,
                total_gates,
                med * 1e3,
                total_gates as f64 / med / 1e3,
                peak_phase_terms
            );
        }
        println!(
            "bench=engine engine={} TOTAL gates={} secs={:.4} kgates_per_s={:.1}",
            stringify!($engine),
            grand_total_gates,
            grand_total_secs,
            grand_total_gates as f64 / grand_total_secs / 1e3
        );
    }};
}

macro_rules! ffi_bench {
    ($engine:ident, $tag:expr, $psum:ident, $zero:ident, $gate:ident, $no_reduce:ident, $cx:ident, $rz:ident) => {{
        use egglog::pathsum::*;
        let mut grand_total_gates = 0usize;
        let mut grand_total_secs = 0.0f64;
        for n in 2..=10usize {
            let windows = gen_windows(n, $tag);
            // Two passes per window: pass 1 is intern-miss dominated,
            // pass 2 (identical states) is intern-hit dominated.
            let total_gates: usize = windows.iter().map(|w| w.len()).sum::<usize>() * 2;
            let run_all = || {
                let mut bv = BaseValues::default();
                bv.register_type::<$psum>();
                for w in &windows {
                    for _pass in 0..2 {
                        let mut v = bv.get($zero(n as i64));
                        for g in w {
                            let state: $psum = bv.unwrap(v);
                            let next = match *g {
                                Gate::X(q) => {
                                    $no_reduce(state, q as i64, |st, q_| st.apply_x(q_))
                                }
                                Gate::Z(q) => $gate(state, q as i64, |st, q_| st.apply_z(q_)),
                                Gate::Sx(q) => {
                                    if state.num_path_vars > PATH_VAR_CAP {
                                        $gate(state, q as i64, |st, q_| st.apply_z(q_))
                                    } else {
                                        $gate(state, q as i64, |st, q_| st.apply_sx(q_))
                                    }
                                }
                                Gate::Rz(q, theta) => {
                                    $rz(state, q as i64, theta.to_bits() as i64)
                                }
                                Gate::Cx(qc, qt) => $cx(state, qc as i64, qt as i64),
                            };
                            v = bv.get(next);
                        }
                    }
                }
            };
            // Warmup
            run_all();
            let mut times = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let start = Instant::now();
                run_all();
                times.push(start.elapsed().as_secs_f64());
            }
            let med = median(times);
            grand_total_gates += total_gates;
            grand_total_secs += med;
            println!(
                "bench=ffi engine={} qubits={} gates={} median_ms={:.3} kgates_per_s={:.1}",
                stringify!($engine),
                n,
                total_gates,
                med * 1e3,
                total_gates as f64 / med / 1e3
            );
        }
        println!(
            "bench=ffi engine={} TOTAL gates={} secs={:.4} kgates_per_s={:.1}",
            stringify!($engine),
            grand_total_gates,
            grand_total_secs,
            grand_total_gates as f64 / grand_total_secs / 1e3
        );
    }};
}

fn main() {
    println!("pathsum_benchmarking seed={SEED:#x} windows={WINDOWS_PER_CONFIG} reps={REPS}");
    engine_bench!(engine_64, 64);
    engine_bench!(engine_128, 128);
    ffi_bench!(
        engine_64,
        64,
        PSum64,
        zero_state_logic_64,
        apply_gate_logic_64,
        apply_gate_no_reduce_logic_64,
        apply_cx_logic_64,
        apply_rz_logic_64
    );
    ffi_bench!(
        engine_128,
        128,
        PSum128,
        zero_state_logic_128,
        apply_gate_logic_128,
        apply_gate_no_reduce_logic_128,
        apply_cx_logic_128,
        apply_rz_logic_128
    );
}
