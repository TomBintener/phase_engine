use egglog::engine_64::*;
fn main() {
    let mut p1 = EvaluatedPathSum::new_id(1);
    p1.apply_sx(0);
    p1.reduce();
    p1.apply_z(0);
    p1.reduce();
    p1.apply_sx(0);
    p1.reduce();
    println!("IZ STATE: {:#?}", p1);
}
