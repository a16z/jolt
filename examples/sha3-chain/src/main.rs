use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha3_chain(target_dir);
    let shared_preprocessing = guest::preprocess_shared_sha3_chain(&mut program);
    let prover_preprocessing = guest::preprocess_prover_sha3_chain(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_sha3_chain(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove_sha3_chain = guest::build_prover_sha3_chain(program, prover_preprocessing);
    let verify_sha3_chain = guest::build_verifier_sha3_chain(verifier_preprocessing);

    let input = [5u8; 32];
    let iters = 100;
    let native_output = guest::sha3_chain(input, iters);
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha3_chain(input, iters);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha3_chain(input, iters, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {}", hex::encode(output));
    info!("native_output: {}", hex::encode(native_output));
    info!("valid: {is_valid}");
}
