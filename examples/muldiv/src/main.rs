use std::time::Instant;

pub fn main() {
    let (prove, verify) = guest::build_muldiv();

    let now = Instant::now();
    let (output, proof) = prove(12031293, 17, 92);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
