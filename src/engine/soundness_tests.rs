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

        /// Interned `Eq` and the dense matrix must agree (up to global phase).
        /// Skips the matrix check when the state is too wide for brute force.
        fn assert_unitarily_equal(a: &EvaluatedPathSum, b: &EvaluatedPathSum) {
            assert_eq!(a, b, "interned Eq failed for unitarily-compared states");
            let n = a.num_qubits as usize;
            let m = a.num_path_vars as usize;
            if n + m > 24 {
                return;
            }
            let ma = pathsum_to_matrix(a);
            let mb = pathsum_to_matrix(b);
            assert!(
                matrices_match_up_to_global_phase(&ma, &mb, TOL),
                "interned-equal states denote different transformations"
            );
        }

        /// If the dense matrices differ, interned `Eq` must not hold.
        /// Skips the matrix check when either state is too wide for brute force;
        /// then only the interned inequality is required.
        fn assert_sound_unequal(a: &EvaluatedPathSum, b: &EvaluatedPathSum) {
            let n = a.num_qubits.max(b.num_qubits) as usize;
            let m = a.num_path_vars.max(b.num_path_vars) as usize;
            if n + m > 24 {
                assert_ne!(a, b, "wide states must not intern-equal when asserted unequal");
                return;
            }
            let ma = pathsum_to_matrix(a);
            let mb = pathsum_to_matrix(b);
            if !matrices_match_up_to_global_phase(&ma, &mb, TOL) {
                assert_ne!(
                    a, b,
                    "unitarily different states intern-equal"
                );
            }
        }

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
                    let live = if state.num_qubits + state.num_path_vars >= <$primitive>::BITS {
                        <$primitive>::MAX
                    } else {
                        (1 as $primitive << (state.num_qubits + state.num_path_vars)) - 1
                    };
                    for &p in &state.continuous_poly.parities {
                        assert_eq!(
                            p & !live,
                            0,
                            "seed {seed}, step {step}: continuous mask has a bit past n+m"
                        );
                    }
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
                assert_sound_unequal(&s1, &s2);
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
            assert_unitarily_equal(&a, &b);

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
            assert_unitarily_equal(&a, &b);
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

        // ------------------------------------------------------------------
        // Canonical gauge + order-independent phase split
        // ------------------------------------------------------------------

        #[derive(Clone, Copy, Debug)]
        enum FuzzGate {
            X(usize),
            Z(usize),
            S(usize),
            Sdg(usize),
            T(usize),
            Tdg(usize),
            Sx(usize),
            H(usize),
            Rz(usize, f64),
            Cx(usize, usize),
        }

        impl FuzzGate {
            fn qubits(self) -> (usize, Option<usize>) {
                match self {
                    FuzzGate::Cx(c, t) => (c, Some(t)),
                    FuzzGate::X(q) | FuzzGate::Z(q) | FuzzGate::S(q) | FuzzGate::Sdg(q)
                    | FuzzGate::T(q) | FuzzGate::Tdg(q) | FuzzGate::Sx(q) | FuzzGate::H(q)
                    | FuzzGate::Rz(q, _) => (q, None),
                }
            }
            fn touches(self, other: FuzzGate) -> bool {
                let (a0, a1) = self.qubits();
                let (b0, b1) = other.qubits();
                let a = [Some(a0), a1];
                let b = [Some(b0), b1];
                a.iter().flatten().any(|x| b.iter().flatten().any(|y| x == y))
            }
        }

        /// Apply one gate to the path sum (with the eager `reduce()` the FFI
        /// performs) and to the dense simulator. `ibm_h` encodes `H` as
        /// `RZ(π/2) SX RZ(π/2)`.
        fn apply_fuzz_gate(state: &mut EvaluatedPathSum, sim: Option<&mut DenseSim>, g: FuzzGate, ibm_h: bool) {
            use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
            let mut sim = sim;
            macro_rules! dense {
                ($e:expr) => { if let Some(s) = sim.as_deref_mut() { $e(s); } };
            }
            match g {
                FuzzGate::X(q) => { state.apply_x(q); dense!(|s: &mut DenseSim| s.x(q)); }
                FuzzGate::Z(q) => { state.apply_z(q); dense!(|s: &mut DenseSim| s.phase_gate(q, PI)); }
                FuzzGate::S(q) => { state.apply_s(q); dense!(|s: &mut DenseSim| s.phase_gate(q, FRAC_PI_2)); }
                FuzzGate::Sdg(q) => { state.apply_sdg(q); dense!(|s: &mut DenseSim| s.phase_gate(q, -FRAC_PI_2)); }
                FuzzGate::T(q) => { state.apply_t(q); dense!(|s: &mut DenseSim| s.phase_gate(q, FRAC_PI_4)); }
                FuzzGate::Tdg(q) => { state.apply_tdg(q); dense!(|s: &mut DenseSim| s.phase_gate(q, -FRAC_PI_4)); }
                FuzzGate::Sx(q) => { state.apply_sx(q); dense!(|s: &mut DenseSim| s.sx(q)); }
                FuzzGate::H(q) => {
                    if ibm_h {
                        state.apply_rz(q, FRAC_PI_2);
                        state.reduce();
                        state.apply_sx(q);
                        state.reduce();
                        state.apply_rz(q, FRAC_PI_2);
                    } else {
                        state.apply_h(q);
                    }
                    dense!(|s: &mut DenseSim| s.h(q));
                }
                FuzzGate::Rz(q, theta) => { state.apply_rz(q, theta); dense!(|s: &mut DenseSim| s.phase_gate(q, theta)); }
                FuzzGate::Cx(c, t) => { state.apply_cx(c, t); dense!(|s: &mut DenseSim| s.cx(c, t)); }
            }
            state.reduce();
        }

        fn run_fuzz_circuit(n: usize, gates: &[FuzzGate], ibm_h: bool) -> (EvaluatedPathSum, DenseSim) {
            let mut state = EvaluatedPathSum::new_id(n as u32);
            let mut sim = DenseSim::identity(n);
            for &g in gates {
                apply_fuzz_gate(&mut state, Some(&mut sim), g, ibm_h);
            }
            (state, sim)
        }

        /// Random circuit over the full gate set with at most `max_hs`
        /// path-variable allocations (H / SX), so brute-force matrices stay small.
        fn random_fuzz_circuit(rng: &mut XorShift, n: usize, len: usize, max_hs: usize) -> Vec<FuzzGate> {
            let mut gates = Vec::with_capacity(len);
            let mut hs = 0usize;
            for _ in 0..len {
                let q = rng.below(n as u64) as usize;
                let g = match rng.below(10) {
                    0 => FuzzGate::X(q),
                    1 => FuzzGate::Z(q),
                    2 => FuzzGate::S(q),
                    3 => FuzzGate::Sdg(q),
                    4 => FuzzGate::T(q),
                    5 => FuzzGate::Tdg(q),
                    6 if hs < max_hs => { hs += 1; FuzzGate::Sx(q) }
                    7 if hs < max_hs => { hs += 1; FuzzGate::H(q) }
                    6 | 7 => FuzzGate::S(q),
                    8 => FuzzGate::Rz(q, FUZZ_ANGLES[rng.below(FUZZ_ANGLES.len() as u64) as usize]),
                    _ => {
                        if n == 1 { FuzzGate::Z(q) } else {
                            let mut c = rng.below(n as u64) as usize;
                            if c == q { c = (c + 1) % n; }
                            FuzzGate::Cx(c, q)
                        }
                    }
                };
                gates.push(g);
            }
            gates
        }

        fn assert_pair(label: &str, n: usize, a: &[FuzzGate], b: &[FuzzGate], ibm_h_b: bool, expect_equal: bool) {
            let (sa, ma) = run_fuzz_circuit(n, a, false);
            let (sb, mb) = run_fuzz_circuit(n, b, ibm_h_b);
            let unitary_equal = matrices_match_up_to_global_phase(&ma.mat, &mb.mat, TOL);
            assert_eq!(unitary_equal, expect_equal, "{label}: dense-matrix premise is wrong");
            assert!(
                matrices_match_up_to_global_phase(&pathsum_to_matrix(&sa), &ma.mat, TOL)
                    && matrices_match_up_to_global_phase(&pathsum_to_matrix(&sb), &mb.mat, TOL),
                "{label}: a path sum diverged from the dense reference"
            );
            assert_eq!(sa == sb, expect_equal, "{label}: interned Eq = {} but unitary equality = {expect_equal}\n a = {sa:?}\n b = {sb:?}", sa == sb);
        }

        /// Operator identities that used to intern unequal because the path
        /// variable gauge (allocation order, H vs SX encoding, affine residue)
        /// or the discrete/continuous split leaked into the canonical form.
        #[test]
        fn test_gauge_and_split_canonical_pairs() {
            use FuzzGate::*;
            use std::f64::consts::{FRAC_PI_4, PI};
            let th = 0.3;
            let ibm_h = |q: usize| -> Vec<FuzzGate> { vec![Rz(q, PI / 2.0), Sx(q), Rz(q, PI / 2.0)] };
            let alt_h = |q: usize| -> Vec<FuzzGate> { vec![Rz(q, PI / 2.0), X(q), Sx(q), X(q), Rz(q, PI / 2.0)] };
            let sxdg = |q: usize| -> Vec<FuzzGate> { vec![X(q), Sx(q)] };
            let cat = |parts: &[&[FuzzGate]]| -> Vec<FuzzGate> { parts.iter().flat_map(|p| p.iter().copied()).collect() };

            // Discrete/continuous split on one parity (H-free).
            assert_pair("RZ T vs RZ(θ+π/4)", 1, &[Rz(0, th), T(0)], &[Rz(0, th + FRAC_PI_4)], false, true);
            assert_pair("T RZ vs RZ(θ+π/4)", 1, &[T(0), Rz(0, th)], &[Rz(0, th + FRAC_PI_4)], false, true);
            assert_pair("CX T CX CX RZ CX vs CX RZ(θ+π/4) CX", 2,
                &[Cx(0, 1), T(1), Cx(0, 1), Cx(0, 1), Rz(1, th), Cx(0, 1)],
                &[Cx(0, 1), Rz(1, th + FRAC_PI_4), Cx(0, 1)], false, true);
            assert_pair("CX RZ Z CX vs CX RZ(θ+π) CX", 2,
                &[Cx(0, 1), Rz(1, th), Z(1), Cx(0, 1)],
                &[Cx(0, 1), Rz(1, th + PI), Cx(0, 1)], false, true);
            assert_pair("Sdg Tdg vs RZ(5π/4)", 1, &[Sdg(0), Tdg(0)], &[Rz(0, 5.0 * FRAC_PI_4)], false, true);
            assert_pair("X RZ vs RZ(-θ) X", 1, &[X(0), Rz(0, th)], &[Rz(0, -th), X(0)], false, true);

            // Path-variable allocation order.
            assert_pair("H0 H1 vs H1 H0", 2, &[H(0), H(1)], &[H(1), H(0)], false, true);
            assert_pair("H0 H1 RZ0 vs H1 H0 RZ0", 2, &[H(0), H(1), Rz(0, th)], &[H(1), H(0), Rz(0, th)], false, true);
            assert_pair("HTH(0) HTH(1) vs HTH(1) HTH(0)", 2,
                &[H(0), T(0), H(0), H(1), T(1), H(1)],
                &[H(1), T(1), H(1), H(0), T(0), H(0)], false, true);
            assert_pair("SX0 SX1 vs SX1 SX0", 2, &[Sx(0), Sx(1)], &[Sx(1), Sx(0)], false, true);

            // Affine residue on a row / H vs SX encodings.
            assert_pair("H X vs Z H", 1, &[H(0), X(0)], &[Z(0), H(0)], false, true);
            assert_pair("H0 H1 CX01 vs CX10 H0 H1", 2, &[H(0), H(1), Cx(0, 1)], &[Cx(1, 0), H(0), H(1)], false, true);
            assert_pair("nam H vs ibm RZ SX RZ", 1, &[H(0)], &ibm_h(0), false, true);
            assert_pair("nam H vs alt RZ X SX X RZ", 1, &[H(0)], &alt_h(0), false, true);
            assert_pair("SX vs H S H", 1, &[Sx(0)], &[H(0), S(0), H(0)], false, true);
            assert_pair("SXdg vs H Sdg H", 1, &sxdg(0), &[H(0), Sdg(0), H(0)], false, true);

            // Conjugated rotations with different conjugator encodings.
            assert_pair("H RZ H vs ibmH RZ ibmH", 1,
                &[H(0), Rz(0, th), H(0)],
                &cat(&[&ibm_h(0), &[Rz(0, th)], &ibm_h(0)]), false, true);
            assert_pair("H RZ H vs S SX RZ SXdg Sdg", 1,
                &[H(0), Rz(0, th), H(0)],
                &cat(&[&[S(0), Sx(0), Rz(0, th)], &sxdg(0), &[Sdg(0)]]), false, true);
            assert_pair("SX RZ SXdg vs HSH RZ HSdgH", 1,
                &cat(&[&[Sx(0), Rz(0, th)], &sxdg(0)]),
                &[H(0), S(0), H(0), Rz(0, th), H(0), Sdg(0), H(0)], false, true);
            assert_pair("XX gadget H0H1 vs H1H0", 2,
                &[H(0), H(1), Cx(0, 1), Rz(1, th), Cx(0, 1), H(0), H(1)],
                &[H(1), H(0), Cx(0, 1), Rz(1, th), Cx(0, 1), H(1), H(0)], false, true);
            assert_pair("nam XX vs ibm XX", 2,
                &[H(0), H(1), Cx(0, 1), Rz(1, th), Cx(0, 1), H(0), H(1)],
                &[H(0), H(1), Cx(0, 1), Rz(1, th), Cx(0, 1), H(0), H(1)], true, true);

            // Negative controls: canonicalization must not over-merge.
            assert_pair("H CX RZ CX H vs CX H RZ H CX", 2,
                &[H(1), Cx(0, 1), Rz(1, th), Cx(0, 1), H(1)],
                &[Cx(0, 1), H(1), Rz(1, th), H(1), Cx(0, 1)], false, false);
            assert_pair("SX vs H Sdg H", 1, &[Sx(0)], &[H(0), Sdg(0), H(0)], false, false);
            assert_pair("RZ(θ) vs RZ(θ+π/4)", 1, &[Rz(0, th)], &[Rz(0, th + FRAC_PI_4)], false, false);
            assert_pair("H0 H1 vs H0", 2, &[H(0), H(1)], &[H(0)], false, false);
        }

        /// Soundness of the canonicalizing `reduce()`: after every gate of a
        /// random circuit the path sum must still denote the dense unitary.
        #[test]
        fn test_gauge_canonicalization_is_sound() {
            let mut rng = XorShift::new(0x6A06_E5C4_0000_0001);
            let mut checks = 0u32;
            for case in 0..2000u32 {
                let n = 1 + rng.below(4) as usize;
                let len = 4 + rng.below(14) as usize;
                let gates = random_fuzz_circuit(&mut rng, n, len, 8);
                let mut state = EvaluatedPathSum::new_id(n as u32);
                let mut sim = DenseSim::identity(n);
                for (i, &g) in gates.iter().enumerate() {
                    apply_fuzz_gate(&mut state, Some(&mut sim), g, false);
                    assert!(
                        matrices_match_up_to_global_phase(&pathsum_to_matrix(&state), &sim.mat, TOL),
                        "case {case}: path sum diverged from dense after gate {i} ({g:?}) of {gates:?}"
                    );
                    // `reduce()` must leave a fixed point of the canonicalization.
                    let mut again = state.clone();
                    assert!(!again.canonicalize_gauge(), "case {case}: canonicalize_gauge is not idempotent after gate {i} of {gates:?}");
                    assert_eq!(again, state);
                    checks += 1;
                }
            }
            assert!(checks > 15000);
        }

        /// Completeness of the canonical form on three rewrite families that
        /// are equal operators by construction: swapping adjacent gates on
        /// disjoint qubits, re-encoding every `H` as `RZ(π/2) SX RZ(π/2)`, and
        /// pushing an `X` through an adjacent `RZ`. Every rewrite must intern
        /// equal to the original.
        #[test]
        fn test_gauge_canonicalization_completeness() {
            let mut rng = XorShift::new(0xC0DE_CAFE_0000_0003);
            let (mut swaps, mut ibm, mut pushes) = (0u32, 0u32, 0u32);
            for case in 0..2000u32 {
                let n = 2 + rng.below(3) as usize;
                let len = 6 + rng.below(10) as usize;
                let gates = random_fuzz_circuit(&mut rng, n, len, 5);
                let (base, base_sim) = run_fuzz_circuit(n, &gates, false);

                // (a) first adjacent pair on disjoint qubits, swapped.
                if let Some(i) = (0..gates.len() - 1).find(|&i| !gates[i].touches(gates[i + 1])) {
                    let mut alt = gates.clone();
                    alt.swap(i, i + 1);
                    let (st, sim) = run_fuzz_circuit(n, &alt, false);
                    assert!(matrices_match_up_to_global_phase(&base_sim.mat, &sim.mat, TOL));
                    assert_eq!(base, st, "case {case}: disjoint swap at {i} changed the canonical form\n {gates:?}");
                    swaps += 1;
                }
                // (b) nam H -> ibm RZ SX RZ.
                if gates.iter().any(|g| matches!(g, FuzzGate::H(_))) {
                    let (st, sim) = run_fuzz_circuit(n, &gates, true);
                    assert!(matrices_match_up_to_global_phase(&base_sim.mat, &sim.mat, TOL));
                    assert_eq!(base, st, "case {case}: ibm H encoding changed the canonical form\n {gates:?}");
                    ibm += 1;
                }
                // (c) X RZ(θ) -> RZ(-θ) X on the same qubit.
                if let Some(i) = (0..gates.len() - 1).find(|&i| matches!((gates[i], gates[i + 1]), (FuzzGate::X(q), FuzzGate::Rz(r, _)) if q == r)) {
                    let mut alt = gates.clone();
                    if let (FuzzGate::X(q), FuzzGate::Rz(_, th)) = (gates[i], gates[i + 1]) {
                        alt[i] = FuzzGate::Rz(q, -th);
                        alt[i + 1] = FuzzGate::X(q);
                    }
                    let (st, sim) = run_fuzz_circuit(n, &alt, false);
                    assert!(matrices_match_up_to_global_phase(&base_sim.mat, &sim.mat, TOL));
                    assert_eq!(base, st, "case {case}: X-through-RZ push changed the canonical form\n {gates:?}");
                    pushes += 1;
                }
            }
            assert!(swaps > 1500 && ibm > 1000 && pushes > 30, "fuzz families under-sampled: {swaps} {ibm} {pushes}");
        }

        /// `apply_phase(p, θ)` and `apply_phase(p ⊕ 1, −θ)` must intern equal.
        #[test]
        fn test_continuous_constant_fold_differential() {
            let mut rng = XorShift::new(0xA11C_E555_C0DE_0001 | 1);
            for case in 0..500u32 {
                // Linear XOR of a few low bits; product monomials are not a mask.
                let mut mask = rng.below(16) as $primitive;
                let mut terms = smallvec::SmallVec::new();
                while mask != 0 {
                    let bit = mask.trailing_zeros();
                    terms.push(1 as $primitive << bit);
                    mask &= mask - 1;
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
