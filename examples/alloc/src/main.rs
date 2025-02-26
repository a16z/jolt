use std::time::Instant;

pub fn main() {
    let (prove_alloc, verify_alloc) = guest::build_alloc();

    let now = Instant::now();
    let (output, proof) = prove_alloc(41);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_alloc(proof);

    println!("output: {:?}", output);
    println!("valid: {}", is_valid);
}
