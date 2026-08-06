use egglog::engine_64::*;
fn main() {
    let t1 = PackedPhaseTerm::create(1, 4);
    let t2 = PackedPhaseTerm::create(0, 2);
    let t3 = PackedPhaseTerm::create(0, 1);
    println!("t1: {:?}", t1);
    println!("t2: {:?}", t2);
    println!("t3: {:?}", t3);
}
