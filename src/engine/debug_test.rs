#[cfg(test)]
mod debug_tests {
    use crate::engine::engine_64;
    use crate::pathsum::synthesize_steiner_logic_64;
    use std::f64::consts::PI;
    use egglog_core_relations::Boxed;

    #[test]
    fn test_debug_pathsum() {
        println!("DEBUGGING PATHSUM EVALUATION...");
        
        let mut state1 = engine_64::EvaluatedPathSum::new_id(2);
        state1.apply_cx(0, 1);
        state1.apply_rz(1, PI / 4.0);
        state1.apply_cx(0, 1);
        
        println!("Circuit 1 (cx; rz; cx):");
        println!("  Continuous Parities: {:?}", state1.continuous_poly.parities);
        println!("  Continuous Phases: {:?}", state1.continuous_poly.phases);
        print!("  Phase Poly Terms: ");
        for term in &state1.phase_poly.terms {
            let phase_unit = term.0 >> 61;
            let mask = term.0 & 0x1FFFFFFFFFFFFFFF;
            print!("(mask={}, phase_unit={}) ", mask, phase_unit);
        }
        println!();
        let p_box1 = egglog_core_relations::Boxed(std::sync::Arc::new(state1.clone()));
        println!("  Synthesized Steiner: {}", synthesize_steiner_logic_64(p_box1.clone(), 10, 0, "".to_string(), 1, 1));
        println!("  Synthesized PMH: {}", crate::pathsum::synthesize_pmh_logic_64(p_box1, 10));

        let mut state2 = engine_64::EvaluatedPathSum::new_id(2);
        state2.apply_rz(0, 3.0 * PI / 4.0);
        state2.apply_rz(1, 3.0 * PI / 4.0);
        
        println!("Circuit 2 (rz 0; rz 1):");
        println!("  Continuous Parities: {:?}", state2.continuous_poly.parities);
        println!("  Continuous Phases: {:?}", state2.continuous_poly.phases);
        print!("  Phase Poly Terms: ");
        for term in &state2.phase_poly.terms {
            let phase_unit = term.0 >> 61;
            let mask = term.0 & 0x1FFFFFFFFFFFFFFF;
            print!("(mask={}, phase_unit={}) ", mask, phase_unit);
        }
        println!();
        let p_box2 = egglog_core_relations::Boxed(std::sync::Arc::new(state2.clone()));
        println!("  Synthesized Steiner: {}", synthesize_steiner_logic_64(p_box2.clone(), 10, 0, "".to_string(), 1, 1));
        println!("  Synthesized PMH: {}", crate::pathsum::synthesize_pmh_logic_64(p_box2, 10));
        
        println!("Are they equal? {}", state1 == state2);
        assert_ne!(state1, state2);
    }
}
