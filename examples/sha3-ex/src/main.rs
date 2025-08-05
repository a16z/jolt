use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha3(target_dir);

    let prover_preprocessing = guest::preprocess_prover_sha3(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_sha3(&prover_preprocessing);

    let prove_sha3 = guest::build_prover_sha3(program, prover_preprocessing);
    let verify_sha3 = guest::build_verifier_sha3(verifier_preprocessing);

    let input = b"Hello, world!";
    let native_output = guest::sha3(input);
    let now = Instant::now();
    let (output, proof) = prove_sha3(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(input, output, proof);

    assert_eq!(output, native_output, "output mismatch");
    println!("output: {}", hex::encode(output));
    println!("native_output: {}", hex::encode(native_output));
    println!("valid: {is_valid}");
}
