use std::time::Instant;

pub fn main() {
    let (prove_sha3, verify_sha3) = guest::build_sha3();

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof) = prove_sha3(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {}", is_valid);
}
