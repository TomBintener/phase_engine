use engine_64::*;
use continuous_poly::*;
use canonical_phase_poly::*;

#[test]
fn debug_pathsum() {
    let mut state = PSum64::id_pathsum(2);
    state.apply_cx(0, 1);
    state.apply_rz(1, std::f64::consts::PI / 4.0);
    state.apply_cx(0, 1);
    
    println!("Continuous Parities: {:?}", state.continuous_poly.parities);
    println!("Continuous Phases: {:?}", state.continuous_poly.phases);
    
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 61;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFF;
        println!("Phase Poly: mask={}, phase_unit={}", mask, phase_unit);
    }
}
