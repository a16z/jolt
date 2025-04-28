use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_sha3_chain(target_dir);

    let prover_preprocessing = guest::preprocess_prover_sha3_chain(&program);
    let verifier_preprocessing = guest::preprocess_verifier_sha3_chain(&program);

    let prove_sha3_chain = guest::build_prover_sha3_chain(program, prover_preprocessing);
    let verify_sha3_chain = guest::build_verifier_sha3_chain(verifier_preprocessing);

    let input = [5u8; 32];
    let iters = 100;
    let now = Instant::now();
    let (output, proof) = prove_sha3_chain(input, iters);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3_chain(input, iters, output, proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {is_valid}");
}
