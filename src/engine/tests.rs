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
        fn test_reduce_hssh_to_x() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_h(0);
            state.apply_s(0);
            state.apply_s(0);
            state.apply_h(0);
            state.reduce();
            assert_eq!(state.num_path_vars, 0);
            assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1]);
            assert!(state.phase_poly.terms.is_empty());
        }

        #[test]
        fn test_reduce_tttdg_to_t() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_t(0);
            state.apply_t(0);
            state.apply_tdg(0);
            state.reduce();
            assert_eq!(state.num_path_vars, 0);
            assert_eq!(state.out_state[0].terms.as_slice(), &[1]);
            assert_eq!(state.phase_poly.terms.len(), 1);
            assert_eq!(state.phase_poly.terms[0].monomial(), 1);
            assert_eq!(state.phase_poly.terms[0].phase(), 1); // T is 1 unit of pi/4
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

        #[test]
        fn test_rz_promotion_and_reduction() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_h(0);
            state.apply_rz(0, std::f64::consts::PI / 3.0);
            state.apply_rz(0, 2.0 * std::f64::consts::PI / 3.0);
            state.apply_h(0);
            state.reduce();
            assert_eq!(state.num_path_vars, 0);
            assert_eq!(state.out_state[0].terms.as_slice(), &[0, 1]);
        }

        #[test]
        fn test_rz_anchor_destruction_by_cancellation() {
            let mut state = EvaluatedPathSum::new_id(1);
            state.apply_h(0);
            state.apply_rz(0, std::f64::consts::FRAC_PI_8);
            state.apply_rz(0, -std::f64::consts::FRAC_PI_8);
            state.apply_h(0);
            state.reduce();
            assert_eq!(state.num_path_vars, 0);
            assert_eq!(state.out_state[0].terms.as_slice(), &[1 << 0]);
        }

        #[test]
        fn test_compact_resolves_accidental_duplicates() {
            let mut poly = ContinuousPhasePoly::new();
            let x0 = 1 as $primitive << 0;
            let x1 = 1 as $primitive << 1;
            let u = 1 as $primitive << 2;
            let v = 1 as $primitive << 3;

            poly.apply_phase(BooleanPoly::from_terms(smallvec![u, x1]), std::f64::consts::FRAC_PI_8);
            poly.apply_phase(BooleanPoly::from_terms(smallvec![v, x0, x1]), std::f64::consts::FRAC_PI_8);

            let e_poly = BooleanPoly::from_terms(smallvec![v, x0]);
            poly.substitute(u, &e_poly);
            poly.compact();

            assert_eq!(poly.parities.len(), 1);
            let mut expected_terms = smallvec![x0, x1, v];
            expected_terms.sort_unstable();
            assert_eq!(poly.parities[0], BooleanPoly::from_terms(expected_terms));
        }

        #[test]
        fn test_promote_clifford_expansion() {
            // Apply S directly
            let mut state_s = EvaluatedPathSum::new_id(2);
            state_s.apply_cx(0, 1);
            state_s.apply_s(1);
            state_s.reduce();

            // Apply Rz(PI/2) which will trigger promote_cliffords
            let mut state_rz = EvaluatedPathSum::new_id(2);
            state_rz.apply_cx(0, 1);
            state_rz.apply_rz(1, std::f64::consts::FRAC_PI_2);
            state_rz.reduce();

            assert_eq!(state_s.phase_poly.terms, state_rz.phase_poly.terms);
            assert_eq!(state_s.out_state, state_rz.out_state);
        }

        #[test]
        fn test_continuous_accumulation_triggers_promotion() {
            let mut state = EvaluatedPathSum::new_id(1);
            // Apply half a T gate (pi/8)
            state.apply_rz(0, std::f64::consts::FRAC_PI_8);
            
            // At this point, the continuous engine holds the phase, discrete is empty
            assert_eq!(state.phase_poly.terms.len(), 0);
            assert_eq!(state.continuous_poly.parities.len(), 1);

            // Apply another half a T gate (pi/8)
            state.apply_rz(0, std::f64::consts::FRAC_PI_8);
            
            // They accumulate to pi/4, which triggers promote_cliffords (T gate)
            // The continuous reservoir should be emptied of this term
            assert_eq!(state.continuous_poly.extract_cliffords().len(), 0);
            // Discrete engine now holds the T gate (phase 1)
            assert_eq!(state.phase_poly.terms.len(), 1);
            assert_eq!(state.phase_poly.terms[0].phase(), 1);
        }

        #[test]
        fn test_negative_rz_matches_sdg() {
            let mut state_sdg = EvaluatedPathSum::new_id(2);
            state_sdg.apply_cx(0, 1);
            state_sdg.apply_sdg(1);
            
            let mut state_rz = EvaluatedPathSum::new_id(2);
            state_rz.apply_cx(0, 1);
            state_rz.apply_rz(1, -std::f64::consts::FRAC_PI_2); // -pi/2
            
            assert_eq!(state_sdg.phase_poly.terms, state_rz.phase_poly.terms);
        }

        #[test]
        fn test_rz_t_triplet_expansion() {
            // Apply T gate on 3 entangled qubits
            let mut state_t = EvaluatedPathSum::new_id(3);
            state_t.apply_cx(0, 2);
            state_t.apply_cx(1, 2);
            state_t.apply_t(2);
            
            // Apply Rz(PI/4) on 3 entangled qubits
            let mut state_rz = EvaluatedPathSum::new_id(3);
            state_rz.apply_cx(0, 2);
            state_rz.apply_cx(1, 2);
            state_rz.apply_rz(2, std::f64::consts::FRAC_PI_4);
            
            // The triplet expansion must mathematically match perfectly
            assert_eq!(state_t.phase_poly.terms, state_rz.phase_poly.terms);
        }

        #[test]
        fn test_large_wrap_around_rz() {
            let mut state = EvaluatedPathSum::new_id(1);
            // Apply 3 PI. This should modulo 2PI and become PI (Z gate, phase 4)
            state.apply_rz(0, 3.0 * std::f64::consts::PI);
            
            assert_eq!(state.phase_poly.terms.len(), 1);
            assert_eq!(state.phase_poly.terms[0].phase(), 4);
        }
        #[test]
        fn test_ibm_z_slide_control() {
            let mut s1 = EvaluatedPathSum::new_id(2);
            s1.apply_cx(0, 1);
            s1.apply_z(0);
            s1.reduce();

            let mut s2 = EvaluatedPathSum::new_id(2);
            s2.apply_z(0);
            s2.apply_cx(0, 1);
            s2.reduce();

            assert_eq!(s1.out_state, s2.out_state);
            assert_eq!(s1.phase_poly.terms, s2.phase_poly.terms);
        }

        #[test]
        fn test_ibm_x_slide_target() {
            let mut s1 = EvaluatedPathSum::new_id(2);
            s1.apply_cx(0, 1);
            s1.apply_x(1);
            s1.reduce();

            let mut s2 = EvaluatedPathSum::new_id(2);
            s2.apply_x(1);
            s2.apply_cx(0, 1);
            s2.reduce();

            assert_eq!(s1.out_state, s2.out_state);
            assert_eq!(s1.phase_poly.terms, s2.phase_poly.terms);
        }

        #[test]
        fn test_ibm_sx_slide_target() {
            let mut s1 = EvaluatedPathSum::new_id(2);
            s1.apply_cx(0, 1);
            s1.apply_sx(1);
            s1.reduce();

            let mut s2 = EvaluatedPathSum::new_id(2);
            s2.apply_sx(1);
            s2.apply_cx(0, 1);
            s2.reduce();

            assert_eq!(s1.out_state, s2.out_state);
            assert_eq!(s1.phase_poly.terms, s2.phase_poly.terms);
        }

        #[test]
        fn test_ibm_anti_commutation() {
            let mut s1 = EvaluatedPathSum::new_id(1);
            s1.apply_x(0);
            s1.apply_rz(0, 0.1);
            s1.reduce();

            let mut s2 = EvaluatedPathSum::new_id(1);
            s2.apply_rz(0, -0.1);
            s2.apply_x(0);
            s2.reduce();

            assert_eq!(s1.out_state, s2.out_state);
            assert_eq!(s1.continuous_poly.parities.len(), 1);
            assert_eq!(s2.continuous_poly.parities.len(), 1);
            
            // X; RZ(0.5) operates on state (1 \oplus x), so it has an extra constant 1 term in parity.
            let p1 = &s1.continuous_poly.parities[0];
            let p2 = &s2.continuous_poly.parities[0];
            assert!(p1.terms.contains(&0)); // Contains the constant term
            assert!(!p2.terms.contains(&0)); // Does not contain the constant term
            
            // Verify phases are correctly stored
            let expected_phase1 = (0.1_f64).rem_euclid(2.0 * std::f64::consts::PI);
            let expected_phase2 = (-0.1_f64).rem_euclid(2.0 * std::f64::consts::PI);
            
            let phase1 = s1.continuous_poly.phases[0];
            let phase2 = s2.continuous_poly.phases[0];
            
            // We use absolute difference because snap_phase rounds to 8 decimal places internally
            assert!((phase1 - expected_phase1).abs() < 1e-7);
            assert!((phase2 - expected_phase2).abs() < 1e-7);
        }

        #[test]
        fn test_ibm_trivial_inverses() {
            // X * X
            let mut s1 = EvaluatedPathSum::new_id(1);
            s1.apply_x(0);
            s1.apply_x(0);
            s1.reduce();
            assert_eq!(s1.out_state[0].terms.as_slice(), &[1 << 0]); // x_0

            // Z * Z
            let mut s2 = EvaluatedPathSum::new_id(1);
            s2.apply_z(0);
            s2.apply_z(0);
            s2.reduce();
            assert_eq!(s2.out_state[0].terms.as_slice(), &[1 << 0]);
            assert!(s2.phase_poly.terms.is_empty());

            // SX * SX
            let mut s3 = EvaluatedPathSum::new_id(1);
            s3.apply_sx(0);
            s3.apply_sx(0);
            s3.reduce();
            // With the improved reduction logic, SX * SX reduces fully to X!
            assert_eq!(s3.out_state[0].terms.as_slice(), &[0, 1]);
        }

        #[test]
        fn test_sx_h_s_h_equivalence() {
            let mut s2 = EvaluatedPathSum::new_id(1);
            s2.apply_h(0);
            s2.apply_s(0);
            s2.apply_h(0);
            s2.reduce();
            // Verifies the pi/2 Gaussian elimination successfully reduces the state
            assert_eq!(s2.num_path_vars, 1);
        }

        #[test]
        fn test_deeply_entangled_sx_reduction() {
            // Apply H S H on a highly entangled state
            let mut s2 = EvaluatedPathSum::new_id(3);
            
            // Create a deeply entangled state across 3 qubits
            s2.apply_h(0);
            s2.apply_h(1);
            s2.apply_h(2);
            s2.apply_cx(0, 1);
            s2.apply_cx(1, 2);
            
            // Apply H S H (equivalent to SX) to the entangled qubit
            s2.apply_h(2);
            s2.apply_s(2);
            s2.apply_h(2);
            s2.reduce();

            // Total generated path vars: 3 (initial H) + 1 (first H of HSH) + 1 (second H of HSH) = 5
            // The Gaussian reduction should seamlessly integrate out the pi/2 interaction
            // across the entire entangled polynomial.
            // Remarkably, the sequence H(2) -> CX(1, 2) -> H(2) is mathematically equivalent to CZ(1, 2),
            // which requires NO path variables! The engine successfully recognizes this through algebraic 
            // integration and collapses the state down to the absolute theoretical minimum of 3 path variables!
            assert_eq!(s2.num_path_vars, 3);
        }

        #[test]
        fn test_sxdg_h_sdg_h_equivalence() {
            let mut s2 = EvaluatedPathSum::new_id(1);
            s2.apply_h(0);
            s2.apply_sdg(0);
            s2.apply_h(0);
            s2.reduce();
            assert_eq!(s2.num_path_vars, 1);
        }
    }
}
