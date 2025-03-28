use std::time::Instant;

pub fn main() {
    let (prove_sha2_chain, verify_sha2_chain) = guest::build_sha2_chain();

    let input = [5u8; 32];
    let iters = 100;
    let native_output = guest::sha2_chain(input, iters);
    let now = Instant::now();
    let (output, proof) = prove_sha2_chain(input, iters);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha2_chain(proof);

    assert_eq!(output, native_output, "output mismatch");
    println!("output: {}", hex::encode(output));
    println!("native_output: {}", hex::encode(native_output));
    println!("valid: {}", is_valid);
}
