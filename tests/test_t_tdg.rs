use egglog::engine::engine_64::EvaluatedPathSum;

#[test]
fn test_t_tdg_canonicalization() {
    let mut p1 = EvaluatedPathSum::new_id(1);
    let mut p2 = EvaluatedPathSum::new_id(1);
    
    // Print initial state
    println!("p1 initial: {:?}", p1);
    
    p2.apply_t(0);
    p2.reduce();
    println!("p2 after T: {:?}", p2);
    
    p2.apply_tdg(0);
    p2.reduce();
    println!("p2 after Tdg: {:?}", p2);
    
    assert_eq!(p1, p2, "p1: {:?}\np2: {:?}", p1, p2);
}
