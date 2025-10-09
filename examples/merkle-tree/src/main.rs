use jolt_sdk::{TrustedAdvice, UntrustedAdvice};
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_merkle_tree(target_dir);

    let prover_preprocessing = guest::preprocess_prover_merkle_tree(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_merkle_tree(&prover_preprocessing);

    let prove_merkle_tree = guest::build_prover_merkle_tree(program, prover_preprocessing);
    let verify_merkle_tree = guest::build_verifier_merkle_tree(verifier_preprocessing);

    let first_input: &[u8] = &[5u8; 32];
    let second_input = [6u8; 32];
    let third_input = [7u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) =
        prove_merkle_tree(first_input, TrustedAdvice::new(second_input), UntrustedAdvice::new(third_input));
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_merkle_tree(first_input, output, program_io.panic, proof);

    info!("output: {}", hex::encode(output));
    info!("valid: {is_valid}");
}
