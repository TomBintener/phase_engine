// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro
// (test builds only).
//
// Dense reference semantics: converts an `EvaluatedPathSum` into the explicit
// matrix it denotes by brute-force enumeration of inputs and path variables,
// and provides a gate-by-gate dense simulator to check the symbolic engine
// against. All comparisons are up to global phase, matching the engine's
// convention of dropping global phase terms.

#[allow(unused_macros)]
macro_rules! define_reference_logic {
    (
        $primitive:ty
    ) => {
        #[derive(Clone, Copy, Debug)]
        pub struct C64 { pub re: f64, pub im: f64 }

        impl C64 {
            pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
            pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };

            pub fn from_polar(theta: f64) -> C64 { C64 { re: theta.cos(), im: theta.sin() } }
            pub fn add(self, o: C64) -> C64 { C64 { re: self.re + o.re, im: self.im + o.im } }
            pub fn sub(self, o: C64) -> C64 { C64 { re: self.re - o.re, im: self.im - o.im } }
            pub fn mul(self, o: C64) -> C64 {
                C64 {
                    re: self.re * o.re - self.im * o.im,
                    im: self.re * o.im + self.im * o.re,
                }
            }
            pub fn scale(self, s: f64) -> C64 { C64 { re: self.re * s, im: self.im * s } }
            pub fn conj(self) -> C64 { C64 { re: self.re, im: -self.im } }
            pub fn norm_sq(self) -> f64 { self.re * self.re + self.im * self.im }
        }

        /// Evaluates a boolean polynomial at a full variable assignment
        /// (bit i of `assign` = value of variable i; monomial 0 = constant 1).
        pub fn eval_bool_poly(poly: &BooleanPoly, assign: $primitive) -> bool {
            let mut acc = false;
            for &t in &poly.terms {
                if (assign & t) == t { acc = !acc; }
            }
            acc
        }

        /// Brute-force converts a path sum into the dense matrix it denotes
        /// (row-major, entry [out * dim + in]). Input variables occupy bits
        /// `0..n`, path variables bits `n..n+m`.
        pub fn pathsum_to_matrix(state: &EvaluatedPathSum) -> Vec<C64> {
            let n = state.num_qubits as usize;
            let m = state.num_path_vars as usize;
            assert!(n + m <= 24, "state too large for brute-force enumeration");
            let dim = 1usize << n;
            let norm = 1.0 / ((1u64 << m) as f64).sqrt();
            let mut mat = vec![C64::ZERO; dim * dim];
            for x in 0..dim {
                for v in 0..(1usize << m) {
                    let assign = (x as $primitive) | ((v as $primitive) << n);
                    let mut phase = 0.0f64;
                    for term in &state.phase_poly.terms {
                        let mono = term.monomial();
                        if (assign & mono) == mono {
                            phase += term.phase() as f64 * std::f64::consts::FRAC_PI_4;
                        }
                    }
                    for (parity, ticks) in state
                        .continuous_poly
                        .parities
                        .iter()
                        .zip(state.continuous_poly.phases.iter())
                    {
                        if eval_bool_poly(parity, assign) { phase += ticks_to_angle(*ticks); }
                    }
                    let mut out = 0usize;
                    for q in 0..n {
                        if eval_bool_poly(&state.out_state[q], assign) { out |= 1 << q; }
                    }
                    let idx = out * dim + x;
                    mat[idx] = mat[idx].add(C64::from_polar(phase).scale(norm));
                }
            }
            mat
        }

        /// True when `b == e^{i phi} * a` for some real phi, entrywise within `tol`.
        pub fn matrices_match_up_to_global_phase(a: &[C64], b: &[C64], tol: f64) -> bool {
            if a.len() != b.len() { return false; }
            let mut k = 0usize;
            let mut best = -1.0f64;
            for (i, x) in a.iter().enumerate() {
                let mag = x.norm_sq();
                if mag > best { best = mag; k = i; }
            }
            if best <= tol * tol {
                return b.iter().all(|x| x.norm_sq() <= tol * tol);
            }
            let ratio = b[k].mul(a[k].conj()).scale(1.0 / best);
            if (ratio.norm_sq() - 1.0).abs() > 1e-6 { return false; }
            a.iter().zip(b.iter()).all(|(x, y)| ratio.mul(*x).sub(*y).norm_sq() <= tol * tol)
        }

        /// Dense gate-by-gate simulator building the full unitary via
        /// left-multiplication, used as the ground truth for fuzz tests.
        pub struct DenseSim {
            pub n: usize,
            pub mat: Vec<C64>,
        }

        impl DenseSim {
            pub fn identity(n: usize) -> Self {
                let dim = 1usize << n;
                let mut mat = vec![C64::ZERO; dim * dim];
                for i in 0..dim { mat[i * dim + i] = C64::ONE; }
                Self { n, mat }
            }

            fn apply_1q(&mut self, q: usize, g: [C64; 4]) {
                let dim = 1usize << self.n;
                let bit = 1usize << q;
                for col in 0..dim {
                    for r0 in 0..dim {
                        if (r0 & bit) != 0 { continue; }
                        let r1 = r0 | bit;
                        let a = self.mat[r0 * dim + col];
                        let b = self.mat[r1 * dim + col];
                        self.mat[r0 * dim + col] = g[0].mul(a).add(g[1].mul(b));
                        self.mat[r1 * dim + col] = g[2].mul(a).add(g[3].mul(b));
                    }
                }
            }

            pub fn x(&mut self, q: usize) {
                self.apply_1q(q, [C64::ZERO, C64::ONE, C64::ONE, C64::ZERO]);
            }

            /// diag(1, e^{i theta}) — matches the engine convention for z/s/t/rz.
            pub fn phase_gate(&mut self, q: usize, theta: f64) {
                self.apply_1q(q, [C64::ONE, C64::ZERO, C64::ZERO, C64::from_polar(theta)]);
            }

            pub fn h(&mut self, q: usize) {
                let s = std::f64::consts::FRAC_1_SQRT_2;
                let p = C64 { re: s, im: 0.0 };
                let np = C64 { re: -s, im: 0.0 };
                self.apply_1q(q, [p, p, p, np]);
            }

            /// (I - iX)/sqrt(2) — the engine's SX convention (physical SX up to
            /// a global phase of e^{i pi/4}).
            pub fn sx(&mut self, q: usize) {
                let s = std::f64::consts::FRAC_1_SQRT_2;
                let d = C64 { re: s, im: 0.0 };
                let o = C64 { re: 0.0, im: -s };
                self.apply_1q(q, [d, o, o, d]);
            }

            pub fn cx(&mut self, qc: usize, qt: usize) {
                let dim = 1usize << self.n;
                let cbit = 1usize << qc;
                let tbit = 1usize << qt;
                for col in 0..dim {
                    for r in 0..dim {
                        if (r & cbit) != 0 && (r & tbit) == 0 {
                            let r2 = r | tbit;
                            self.mat.swap(r * dim + col, r2 * dim + col);
                        }
                    }
                }
            }
        }
    }
}
