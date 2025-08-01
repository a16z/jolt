use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha2_chain(target_dir);

    let prover_preprocessing = guest::preprocess_prover_sha2_chain(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_sha2_chain(&prover_preprocessing);

    let prove_sha2_chain = guest::build_prover_sha2_chain(program, prover_preprocessing);
    let verify_sha2_chain = guest::build_verifier_sha2_chain(verifier_preprocessing);

    let input = [5u8; 32];
    let iters = 1000;
    let native_output = guest::sha2_chain(input, iters);
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha2_chain(input, iters);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha2_chain(input, iters, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    println!("output: {}", hex::encode(output));
    println!("native_output: {}", hex::encode(native_output));
    println!("valid: {is_valid}");
}
