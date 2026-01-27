use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha3(target_dir);
    let prover_preprocessing = guest::preprocess_sha3(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_sha3(&prover_preprocessing);

    let prove_sha3 = guest::build_prover_sha3(program, prover_preprocessing);
    let verify_sha3 = guest::build_verifier_sha3(verifier_preprocessing);

    let input = b"Hello, world!";
    let native_output = guest::sha3(input);
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha3(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3(input, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {}", hex::encode(output));
    info!("native_output: {}", hex::encode(native_output));
    info!("valid: {is_valid}");
}
