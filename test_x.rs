use egglog::engine_64::*;
fn main() {
    let mut p1 = EvaluatedPathSum::new_id(1);
    p1.apply_sx(0);
    p1.reduce();
    p1.apply_sx(0);
    p1.reduce();
    
    let mut p2 = EvaluatedPathSum::new_id(1);
    p2.apply_x(0);
    p2.reduce();
    
    println!("SX SX STATE: {:#?}", p1);
    println!("X STATE: {:#?}", p2);
}
