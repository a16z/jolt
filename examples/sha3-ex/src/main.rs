use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_sha3(target_dir);

    let prover_preprocessing = guest::preprocess_prover_sha3(&program);
    let verifier_preprocessing = guest::preprocess_verifier_sha3(&program);

    let prove_sha3 = guest::build_prover_sha3(program, prover_preprocessing);
    let verify_sha3 = guest::build_verifier_sha3(verifier_preprocessing);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof) = prove_sha3(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(input, output, proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {is_valid}");
}
