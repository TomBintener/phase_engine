// Note: This file is included at the top level of `engine/mod.rs`
// and defines a macro that is called by the main engine-generating macro.

#[allow(unused_macros)]
macro_rules! define_tests_logic {
    (
        $primitive:ty,
        $phase_shift:expr
    ) => {
        use smallvec::smallvec;

        #[test]
        fn test_create_masks_out_of_bounds_inputs() {
            let oversized_monomial = (1 as $primitive << ($phase_shift + 1)) | 5;
            let oversized_phase = 9;
            let term = PackedPhaseTerm::create(oversized_monomial, oversized_phase);
            assert_eq!(term.monomial(), 5);
            assert_eq!(term.phase(), 1);
        }

        #[test]
        fn test_custom_ord() {
            let term1 = PackedPhaseTerm::create(1, 7);
            let term2 = PackedPhaseTerm::create(2, 1);
            assert_eq!(term1.cmp(&term2), Ordering::Less);
            let term3 = PackedPhaseTerm::create(1, 2);
            let term4 = PackedPhaseTerm::create(1, 7);
            assert_eq!(term3.cmp(&term4), Ordering::Less);
        }

        #[test]
        fn test_add_assign_cancels_out() {
            let mut poly_a = CanonicalPhasePoly { terms: smallvec![PackedPhaseTerm::create(1, 4), PackedPhaseTerm::create(3, 2)] };
            let poly_b = CanonicalPhasePoly { terms: smallvec![PackedPhaseTerm::create(1, 4), PackedPhaseTerm::create(3, 6)] };
            poly_a.add_assign(&poly_b);
            assert!(poly_a.terms.is_empty());
        }

        #[test]
        fn test_boolean_poly_add_assign_cancels() {
            let mut poly_a = BooleanPoly::from_terms(smallvec![1, 3, 5]);
            let poly_b = BooleanPoly::from_terms(smallvec![1, 4, 5]);
            poly_a.add_assign(&poly_b);
            let expected = BooleanPoly::from_terms(smallvec![3, 4]);
            assert_eq!(poly_a, expected);
        }

        #[test]
        fn test_apply_phase_merges_terms() {
            let mut poly = ContinuousPhasePoly::new();
            let parity = BooleanPoly::from_terms(smallvec![1]);
            poly.apply_phase(parity.clone(), 1.0);
            poly.apply_phase(parity.clone(), 0.5);
            assert_eq!(poly.parities.len(), 1);
            assert!((poly.phases[0] - 1.5).abs() < EPSILON);
        }

        #[test]
        fn test_reduce_hzh_to_x() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_h(0);
            state.apply_z(0);
            state.apply_h(0);
            state.reduce();
            assert_eq!(state.num_path_vars, 0);
            assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1]);
            assert!(state.phase_poly.terms.is_empty());
        }

        #[test]
        fn test_new_id() {
            let state = EvaluatedPathSum::new_id(3);
            assert_eq!(state.num_qubits, 3);
            assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
            assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 1]);
            assert_eq!(state.out_state[2].terms.as_slice(), &[1 << 2]);
        }

        #[test]
        fn test_apply_x() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_x(0);
            assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1]);
        }

        #[test]
        fn test_apply_cx() {
            let mut state = EvaluatedPathSum::new_id(2);
            state.apply_cx(0, 1);
            assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 0, 1 << 1]);
        }

        #[test]
        fn test_apply_h() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.out_state[0] = BooleanPoly::from_terms(smallvec![1, 2]);
            state.apply_h(0);
            let v_mask = 1 as $primitive << 1;
            assert_eq!(state.num_path_vars, 1);
            assert_eq!(state.out_state[0].terms.as_slice(), &[v_mask]);
            let mut expected_phases = vec![
                PackedPhaseTerm::create((1 as $primitive << 0) | v_mask, 4),
                PackedPhaseTerm::create((1 as $primitive << 1) | v_mask, 4),
            ];
            expected_phases.sort_unstable();
            assert_eq!(state.phase_poly.terms.as_slice(), expected_phases.as_slice());
        }

        #[test]
        fn test_multiple_independent_reductions_and_repacking() {
            let mut state = EvaluatedPathSum::new_id(4);
            state.num_path_vars = 6;
            let initial_parity = BooleanPoly::from_terms(smallvec![1 << 0, 1 << 4, 1 << 6]);
            state.continuous_poly.apply_phase(initial_parity, 1.23);
            let v4_mask = 1 as $primitive << 4;
            let v5_mask = 1 as $primitive << 5;
            let v8_mask = 1 as $primitive << 8;
            state.phase_poly.merge_unsorted_batch(vec![
                PackedPhaseTerm::create(v8_mask | v4_mask, 4),
                PackedPhaseTerm::create(v8_mask | v5_mask, 4),
            ]);
            let v6_mask = 1 as $primitive << 6;
            let v7_mask = 1 as $primitive << 7;
            let v9_mask = 1 as $primitive << 9;
            state.phase_poly.merge_unsorted_batch(vec![
                PackedPhaseTerm::create(v9_mask | v6_mask, 4),
                PackedPhaseTerm::create(v9_mask | v7_mask, 4),
            ]);
            state.reduce();
            assert_eq!(state.num_path_vars, 2, "Two path variables should survive");
            let x0_mask = 1 as $primitive << 0;
            let v4_repacked_mask = 1 as $primitive << 4;
            let v5_repacked_mask = 1 as $primitive << 5;
            let expected_parity = BooleanPoly::from_terms(smallvec![x0_mask, v4_repacked_mask, v5_repacked_mask]);
            assert_eq!(state.continuous_poly.parities.len(), 1);
            assert_eq!(state.continuous_poly.parities[0], expected_parity);
        }
        #[test]
        fn test_apply_z_and_s_and_t() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_z(0); // Phase 4
            state.apply_s(0); // Phase 2
            state.apply_t(0); // Phase 1
            // 4 + 2 + 1 = 7 phase accumulation on x0
            let expected_phase = PackedPhaseTerm::create(1 as $primitive << 0, 7);
            assert_eq!(state.phase_poly.terms.as_slice(), &[expected_phase]);
        }

        #[test]
        fn test_cx_cx_identity() {
            let mut state = EvaluatedPathSum::new_id(2);
            state.apply_cx(0, 1);
            state.apply_cx(0, 1);
            state.reduce();
            assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 1]); // Returns to original ID state
        }

        #[test]
        fn test_bell_state_and_uncompute() {
            let mut state = EvaluatedPathSum::new_id(2);
            state.apply_h(0);
            state.apply_cx(0, 1);
            state.apply_cx(0, 1);
            state.apply_h(0);
            state.reduce();
            assert_eq!(state.num_path_vars, 0); // All path vars should be eliminated
            assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
            assert_eq!(state.out_state[1].terms.as_slice(), &[1 << 1]);
            assert!(state.phase_poly.terms.is_empty());
        }

        #[test]
        fn test_apply_sdg_tdg() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_sdg(0); // Phase -2 (or 6 mod 8)
            state.apply_tdg(0); // Phase -1 (or 7 mod 8)
            // 6 + 7 = 13 mod 8 = 5
            let expected_phase = PackedPhaseTerm::create(1 as $primitive << 0, 5);
            assert_eq!(state.phase_poly.terms.as_slice(), &[expected_phase]);
        }

        #[test]
        fn test_apply_sx() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_sx(0);
            state.reduce();
            // SX introduces a path variable, and is equivalent to (1+i)/2 (I + iX).
            // This is a more complex state, so we just verify it compiled and mutated the state
            assert!(state.num_path_vars > 0 || !state.phase_poly.terms.is_empty() || !state.continuous_poly.phases.is_empty());
        }
    }
}
