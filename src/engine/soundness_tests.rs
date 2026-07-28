// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro
// (test builds only).
//
// Soundness tests: verify that gate application and `reduce()` preserve the
// linear map denoted by an `EvaluatedPathSum`, using the dense reference
// simulator from `reference.rs`. These guard the E-graph against unsound
// merges: two path sums must compare equal only when they denote the same
// transformation up to global phase.

#[allow(unused_macros)]
macro_rules! define_soundness_tests_logic {
    (
        $primitive:ty
    ) => {
        const TOL: f64 = 1e-6;

        struct XorShift(u64);

        impl XorShift {
            fn new(seed: u64) -> Self { Self(seed | 1) }
            fn next(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }
            fn below(&mut self, n: u64) -> u64 { self.next() % n }
        }

        const FUZZ_ANGLES: [f64; 10] = [
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

        /// Applies one random gate from the engine's supported set to both the
        /// path sum and the dense reference simulator.
        fn apply_random_gate(rng: &mut XorShift, state: &mut EvaluatedPathSum, sim: &mut DenseSim) {
            let n = state.num_qubits as usize;
            let q = rng.below(n as u64) as usize;
            match rng.below(10) {
                0 => { state.apply_x(q); sim.x(q); }
                1 => { state.apply_z(q); sim.phase_gate(q, std::f64::consts::PI); }
                2 => { state.apply_s(q); sim.phase_gate(q, std::f64::consts::FRAC_PI_2); }
                3 => { state.apply_sdg(q); sim.phase_gate(q, -std::f64::consts::FRAC_PI_2); }
                4 => { state.apply_t(q); sim.phase_gate(q, std::f64::consts::FRAC_PI_4); }
                5 => { state.apply_tdg(q); sim.phase_gate(q, -std::f64::consts::FRAC_PI_4); }
                6 => { state.apply_sx(q); sim.sx(q); }
                7 => { state.apply_h(q); sim.h(q); }
                8 => {
                    let theta = FUZZ_ANGLES[rng.below(FUZZ_ANGLES.len() as u64) as usize];
                    state.apply_rz(q, theta);
                    sim.phase_gate(q, theta);
                }
                _ => {
                    let mut qc = rng.below(n as u64) as usize;
                    if qc == q { qc = (qc + 1) % n; }
                    state.apply_cx(qc, q);
                    sim.cx(qc, q);
                }
            }
        }

        /// Regression for the pivot-substitution unsoundness: `reduce()` pins
        /// `u1 := 1 XOR u2` (a multi-term pivot expression) while an S†-phase
        /// term `6 * u1` is still present. The old code distributed the
        /// substitution linearly over E's terms, which is only valid for pi
        /// coefficients, and corrupted the phase by `pi * u2` here.
        #[test]
        fn test_pivot_substitution_preserves_semantics() {
            let x0 = 1 as $primitive << 0;
            let v  = 1 as $primitive << 1;
            let u1 = 1 as $primitive << 2;
            let u2 = 1 as $primitive << 3;

            let mut state = EvaluatedPathSum::new_id(1);
            state.num_path_vars = 3;
            state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![x0, u1]);
            state.phase_poly.merge_unsorted_batch(vec![
                PackedPhaseTerm::create(v, 4),
                PackedPhaseTerm::create(v | u1, 4),
                PackedPhaseTerm::create(v | u2, 4),
                PackedPhaseTerm::create(u1, 6),
            ]);

            let before = pathsum_to_matrix(&state);
            state.reduce();
            let after = pathsum_to_matrix(&state);
            assert!(
                matrices_match_up_to_global_phase(&before, &after, TOL),
                "pivot substitution changed the denoted transformation"
            );
        }

        /// Regression for the GF(2) parity bug in `BooleanPoly::from_terms`:
        /// eliminating `v` pins `u := e1 XOR e2`; substituting that into the
        /// out_state monomial `u*e1*e2` distributes to `[e1*e2, e1*e2]`, which
        /// must cancel to zero under XOR semantics. The old `dedup()` kept one
        /// copy of the duplicated monomial, corrupting the denoted basis map.
        #[test]
        fn test_pivot_substitution_gf2_parity_cancellation() {
            let x0 = 1 as $primitive << 0;
            let v  = 1 as $primitive << 1;
            let u  = 1 as $primitive << 2;
            let e1 = 1 as $primitive << 3;
            let e2 = 1 as $primitive << 4;

            let mut state = EvaluatedPathSum::new_id(1);
            state.num_path_vars = 4;
            state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![x0, u | e1 | e2]);
            state.phase_poly.merge_unsorted_batch(vec![
                PackedPhaseTerm::create(v | u, 4),
                PackedPhaseTerm::create(v | e1, 4),
                PackedPhaseTerm::create(v | e2, 4),
            ]);

            let before = pathsum_to_matrix(&state);
            state.reduce();
            let after = pathsum_to_matrix(&state);
            assert!(
                matrices_match_up_to_global_phase(&before, &after, TOL),
                "duplicated monomials produced by pivot substitution must cancel (GF(2) parity)"
            );
        }

        /// Regression for the stale liveness-mask unsoundness: eliminating `v`
        /// pins `u := w`, which makes `w` live inside a continuous phase parity.
        /// The old code kept judging liveness against the mask snapshot taken at
        /// the start of the pass, so `w` was subsequently integrated out even
        /// though the continuous polynomial still depended on it.
        #[test]
        fn test_substituted_variables_stay_live() {
            let x0 = 1 as $primitive << 0;
            let v  = 1 as $primitive << 1;
            let u  = 1 as $primitive << 2;
            let w  = 1 as $primitive << 3;
            let y  = 1 as $primitive << 4;

            let mut state = EvaluatedPathSum::new_id(1);
            state.num_path_vars = 4;
            state.out_state[0] = BooleanPoly::from_terms(smallvec::smallvec![x0, y]);
            state.phase_poly.merge_unsorted_batch(vec![
                PackedPhaseTerm::create(v | u, 4),
                PackedPhaseTerm::create(v | w, 4),
                PackedPhaseTerm::create(w | y, 4),
            ]);
            state
                .continuous_poly
                .apply_phase(BooleanPoly::from_terms(smallvec::smallvec![x0, u]), 0.3);

            let before = pathsum_to_matrix(&state);
            state.reduce();
            let after = pathsum_to_matrix(&state);
            assert!(
                matrices_match_up_to_global_phase(&before, &after, TOL),
                "a variable made live by substitution was integrated out"
            );
        }

        /// Random circuits, evaluated exactly as production does (eager reduce
        /// after every gate), must track the dense reference at every step.
        #[test]
        fn test_random_circuits_match_dense_reference() {
            for seed in 0..40u64 {
                let mut rng = XorShift::new(0x9E37_79B9_7F4A_7C15 ^ (seed.wrapping_mul(0xA24B_AED4_963E_E407)));
                let n = 3u32;
                let mut state = EvaluatedPathSum::new_id(n);
                let mut sim = DenseSim::identity(n as usize);
                for step in 0..18 {
                    apply_random_gate(&mut rng, &mut state, &mut sim);
                    state.reduce();
                    if state.num_path_vars > 10 { break; }
                    let mat = pathsum_to_matrix(&state);
                    assert!(
                        matrices_match_up_to_global_phase(&mat, &sim.mat, TOL),
                        "seed {seed}, step {step}: path sum diverged from dense reference"
                    );
                }
            }
        }

        /// `reduce()` on a deeply unreduced state (no intermediate reductions)
        /// must preserve the denoted transformation.
        #[test]
        fn test_reduce_preserves_semantics_fuzz() {
            for seed in 0..40u64 {
                let mut rng = XorShift::new(0xDEAD_BEEF_CAFE_F00D ^ (seed.wrapping_mul(0x2545_F491_4F6C_DD1D)));
                let mut state = EvaluatedPathSum::new_id(3);
                let mut sim = DenseSim::identity(3);
                for _ in 0..12 {
                    apply_random_gate(&mut rng, &mut state, &mut sim);
                    if state.num_path_vars >= 8 { break; }
                }
                let before = pathsum_to_matrix(&state);
                assert!(
                    matrices_match_up_to_global_phase(&before, &sim.mat, TOL),
                    "seed {seed}: unreduced path sum diverged from dense reference"
                );
                state.reduce();
                let after = pathsum_to_matrix(&state);
                assert!(
                    matrices_match_up_to_global_phase(&before, &after, TOL),
                    "seed {seed}: reduce() changed the denoted transformation"
                );
            }
        }

        /// E-graph soundness contract: whenever two path sums compare equal,
        /// the transformations they denote must be equal up to global phase.
        #[test]
        fn test_equality_implies_equal_semantics() {
            for seed in 0..60u64 {
                let mut rng = XorShift::new(0x1234_5678_9ABC_DEF1 ^ (seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                let n = 3u32;
                let mut build = |rng: &mut XorShift| {
                    let mut state = EvaluatedPathSum::new_id(n);
                    let mut sim = DenseSim::identity(n as usize);
                    for _ in 0..6 {
                        apply_random_gate(rng, &mut state, &mut sim);
                        state.reduce();
                    }
                    (state, sim)
                };
                let (s1, sim1) = build(&mut rng);
                let (s2, sim2) = build(&mut rng);
                if s1 == s2 {
                    assert!(
                        matrices_match_up_to_global_phase(&sim1.mat, &sim2.mat, TOL),
                        "seed {seed}: equal path sums denote different transformations"
                    );
                }
            }
        }

        /// Sanity: identical circuits must produce Eq-equal canonical states.
        #[test]
        fn test_identical_circuits_compare_equal() {
            let mut rng = XorShift::new(0xF0F0_F0F0_1234_5678);
            let n = 3u32;
            let mut gates: Vec<(u64, usize, usize, f64)> = Vec::new();
            for _ in 0..14 {
                let g = rng.below(10);
                let q = rng.below(n as u64) as usize;
                let mut qc = rng.below(n as u64) as usize;
                if qc == q { qc = (qc + 1) % n as usize; }
                let theta = FUZZ_ANGLES[rng.below(FUZZ_ANGLES.len() as u64) as usize];
                gates.push((g, q, qc, theta));
            }
            let run = |gates: &[(u64, usize, usize, f64)]| {
                let mut state = EvaluatedPathSum::new_id(n);
                for &(g, q, qc, theta) in gates {
                    match g {
                        0 => state.apply_x(q),
                        1 => state.apply_z(q),
                        2 => state.apply_s(q),
                        3 => state.apply_sdg(q),
                        4 => state.apply_t(q),
                        5 => state.apply_tdg(q),
                        6 => state.apply_sx(q),
                        7 => state.apply_h(q),
                        8 => state.apply_rz(q, theta),
                        _ => state.apply_cx(qc, q),
                    }
                    state.reduce();
                }
                state
            };
            assert_eq!(run(&gates), run(&gates));
        }
    }
}
