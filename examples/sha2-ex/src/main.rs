use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha2(target_dir);

    let prover_preprocessing = guest::preprocess_sha2(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_sha2(&prover_preprocessing);

    let prove_sha2 = guest::build_prover_sha2(program, prover_preprocessing);
    let verify_sha2 = guest::build_verifier_sha2(verifier_preprocessing);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha2(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha2(input, output, program_io.panic, proof);

    info!("output: {}", hex::encode(output));
    info!("valid: {is_valid}");
}
