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

        /// Differential test for the linear-merge `add_assign` rewrites: the
        /// merge-based implementation must produce exactly the same canonical
        /// term vectors as a naive extend+sort+compact oracle, on random
        /// polynomial pairs — including `other` inputs that carry within-run
        /// duplicates (as `ContinuousPhasePoly::substitute` produces).
        #[test]
        fn test_add_assign_matches_naive_oracle() {
            let mut rng = XorShift::new(0xA55E_55ED_0DDB_A11 | 1);

            // --- BooleanPoly ---
            for case in 0..500u32 {
                // Canonical `self`: via from_terms (sorted, parity-compacted).
                // The SmallVec capacity parameter differs per engine, so the
                // concrete type is inferred from the BooleanPoly signatures.
                let self_len = rng.below(12) as usize;
                let mut self_terms = smallvec::SmallVec::new();
                for _ in 0..self_len {
                    self_terms.push(rng.below(64) as $primitive);
                }
                let base = BooleanPoly::from_terms(self_terms);

                // `other`: sorted but NOT compacted — may contain duplicates.
                let other_len = rng.below(12) as usize;
                let mut other = BooleanPoly::from_terms(smallvec::smallvec![]);
                for _ in 0..other_len {
                    // Small domain to force frequent duplicates.
                    other.terms.push(rng.below(8) as $primitive);
                }
                other.terms.sort_unstable();
                other.variable_mask = other.terms.iter().fold(0, |acc, &x| acc | x);
                let other_terms: Vec<$primitive> = other.terms.to_vec();

                // Oracle: naive extend + sort + whole-run parity compaction.
                let mut oracle: Vec<$primitive> = base.terms.to_vec();
                oracle.extend_from_slice(&other_terms);
                oracle.sort_unstable();
                let mut compacted: Vec<$primitive> = Vec::new();
                let mut i = 0;
                while i < oracle.len() {
                    let mut j = i + 1;
                    while j < oracle.len() && oracle[j] == oracle[i] { j += 1; }
                    if (j - i) % 2 != 0 { compacted.push(oracle[i]); }
                    i = j;
                }

                let mut merged = base.clone();
                merged.add_assign(&other);
                assert_eq!(
                    merged.terms.as_slice(),
                    compacted.as_slice(),
                    "case {case}: BooleanPoly::add_assign diverged from the naive oracle"
                );
                let expect_mask = compacted.iter().fold(0, |acc, &x| acc | x);
                assert_eq!(merged.variable_mask, expect_mask, "case {case}: variable_mask mismatch");
            }

            // --- CanonicalPhasePoly ---
            for case in 0..500u32 {
                let mk_canonical = |rng: &mut XorShift, len_max: u64, dom: u64| {
                    let len = rng.below(len_max) as usize;
                    let mut batch = Vec::with_capacity(len);
                    for _ in 0..len {
                        batch.push(PackedPhaseTerm::create(
                            rng.below(dom) as $primitive,
                            (rng.below(8)) as u8,
                        ));
                    }
                    let mut poly = CanonicalPhasePoly { terms: smallvec::smallvec![] };
                    poly.merge_unsorted_batch(batch);
                    poly
                };
                let base = mk_canonical(&mut rng, 14, 64);
                let other = mk_canonical(&mut rng, 14, 16);

                // Oracle: naive extend + sort + whole-run mod-8 compaction,
                // dropping zero phases and global-phase (monomial 0) terms.
                let mut oracle: Vec<PackedPhaseTerm> = base.terms.to_vec();
                oracle.extend_from_slice(&other.terms);
                oracle.sort_unstable();
                let mut compacted: Vec<PackedPhaseTerm> = Vec::new();
                let mut i = 0;
                while i < oracle.len() {
                    let mono = oracle[i].monomial();
                    let mut phase = oracle[i].phase();
                    let mut j = i + 1;
                    while j < oracle.len() && oracle[j].monomial() == mono {
                        phase = (phase + oracle[j].phase()) % 8;
                        j += 1;
                    }
                    if phase != 0 && mono != 0 {
                        compacted.push(PackedPhaseTerm::create(mono, phase));
                    }
                    i = j;
                }

                let mut merged = base.clone();
                merged.add_assign(&other);
                assert_eq!(
                    merged.terms.as_slice(),
                    compacted.as_slice(),
                    "case {case}: CanonicalPhasePoly::add_assign diverged from the naive oracle"
                );
            }
        }

        /// Operator identities that used to be checked only from `|0>` (ket
        /// masks). They must still intern-equal when folded from \(I_n\).
        #[test]
        fn test_operator_identities_from_id() {
            let mut a = EvaluatedPathSum::new_id(2);
            a.apply_x(0);
            a.apply_h(0);
            a.reduce();
            let mut b = EvaluatedPathSum::new_id(2);
            b.apply_h(0);
            b.apply_z(0);
            b.reduce();
            assert_eq!(a, b, "HX and ZH must be Eq-equal from I_n");

            let mut a = EvaluatedPathSum::new_id(2);
            a.apply_h(0);
            a.apply_x(1);
            a.apply_cx(0, 1);
            a.reduce();
            let mut b = EvaluatedPathSum::new_id(2);
            b.apply_h(0);
            b.apply_cx(0, 1);
            b.apply_x(1);
            b.reduce();
            assert_eq!(a, b, "X;CX and CX;X on the target must be Eq-equal from I_n");
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

        #[derive(Clone, Copy)]
        enum HFreeGate {
            X(usize),
            Cx(usize, usize),
            Rz(usize, f64),
        }

        fn push_x_right(gates: &[HFreeGate], n: usize) -> Vec<HFreeGate> {
            let mut pending = vec![false; n];
            let mut out = Vec::new();
            for &g in gates {
                match g {
                    HFreeGate::X(q) => pending[q] ^= true,
                    HFreeGate::Rz(q, theta) => {
                        let t = if pending[q] { -theta } else { theta };
                        out.push(HFreeGate::Rz(q, t));
                    }
                    HFreeGate::Cx(c, t) => {
                        out.push(HFreeGate::Cx(c, t));
                        if pending[c] {
                            pending[t] ^= true;
                        }
                    }
                }
            }
            for q in 0..n {
                if pending[q] {
                    out.push(HFreeGate::X(q));
                }
            }
            out
        }

        fn apply_hfree_gates(
            state: &mut EvaluatedPathSum,
            sim: &mut DenseSim,
            gates: &[HFreeGate],
        ) {
            for &g in gates {
                match g {
                    HFreeGate::X(q) => {
                        state.apply_x(q);
                        sim.x(q);
                    }
                    HFreeGate::Cx(c, t) => {
                        state.apply_cx(c, t);
                        sim.cx(c, t);
                    }
                    HFreeGate::Rz(q, theta) => {
                        state.apply_rz(q, theta);
                        sim.phase_gate(q, theta);
                    }
                }
                state.reduce();
            }
        }

        /// Completeness: commuting every X to the right of an H-free {X,CX,RZ}
        /// circuit must leave interned PathSum Eq, matching the dense simulator.
        #[test]
        fn test_hfree_x_commutation_completeness() {
            let mut case = 0u32;
            for n in 1u32..=4 {
                for seed in 0..200u64 {
                    let mut rng = XorShift::new(
                        0xC0FF_EE00_F00D_0000 ^ n as u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    );
                    let n_us = n as usize;
                    let mut gates = Vec::with_capacity(12);
                    for _ in 0..12 {
                        match rng.below(3) {
                            0 => gates.push(HFreeGate::X(rng.below(n as u64) as usize)),
                            1 if n > 1 => {
                                let c = rng.below(n as u64) as usize;
                                let mut t = rng.below(n as u64) as usize;
                                if t == c {
                                    t = (t + 1) % n_us;
                                }
                                gates.push(HFreeGate::Cx(c, t));
                            }
                            _ => {
                                let q = rng.below(n as u64) as usize;
                                let theta = FUZZ_ANGLES[rng.below(FUZZ_ANGLES.len() as u64) as usize];
                                gates.push(HFreeGate::Rz(q, theta));
                            }
                        }
                    }
                    let rewritten = push_x_right(&gates, n_us);

                    let mut orig = EvaluatedPathSum::new_id(n);
                    let mut orig_sim = DenseSim::identity(n_us);
                    apply_hfree_gates(&mut orig, &mut orig_sim, &gates);

                    let mut rew = EvaluatedPathSum::new_id(n);
                    let mut rew_sim = DenseSim::identity(n_us);
                    apply_hfree_gates(&mut rew, &mut rew_sim, &rewritten);

                    assert_eq!(
                        orig, rew,
                        "n={n} seed={seed} case={case}: H-free X-push PathSums differ"
                    );
                    let orig_mat = pathsum_to_matrix(&orig);
                    let rew_mat = pathsum_to_matrix(&rew);
                    assert!(
                        matrices_match_up_to_global_phase(&orig_mat, &orig_sim.mat, TOL),
                        "n={n} seed={seed}: original path sum diverged from dense"
                    );
                    assert!(
                        matrices_match_up_to_global_phase(&rew_mat, &rew_sim.mat, TOL),
                        "n={n} seed={seed}: rewritten path sum diverged from dense"
                    );
                    assert!(
                        matrices_match_up_to_global_phase(&orig_sim.mat, &rew_sim.mat, TOL),
                        "n={n} seed={seed}: rewritten circuit is not unitarily equal"
                    );
                    case += 1;
                }
            }
        }

        /// `apply_phase(p, θ)` and `apply_phase(p ⊕ 1, −θ)` must intern equal.
        #[test]
        fn test_continuous_constant_fold_differential() {
            let mut rng = XorShift::new(0xA11C_E555_C0DE_0001 | 1);
            for case in 0..500u32 {
                let len = rng.below(8) as usize;
                let mut terms = smallvec::SmallVec::new();
                for _ in 0..len {
                    terms.push(rng.below(32) as $primitive);
                }
                let p = BooleanPoly::from_terms(terms);
                let theta = FUZZ_ANGLES[rng.below(FUZZ_ANGLES.len() as u64) as usize];

                let mut a = ContinuousPhasePoly::new();
                a.apply_phase(p.clone(), theta);

                let mut xor1_terms = p.terms.clone();
                xor1_terms.push(0);
                let p1 = BooleanPoly::from_terms(xor1_terms);
                let mut b = ContinuousPhasePoly::new();
                b.apply_phase(p1, -theta);

                assert_eq!(
                    a, b,
                    "case {case}: apply_phase(p, θ) != apply_phase(p⊕1, −θ)"
                );
            }
        }
    }
}
