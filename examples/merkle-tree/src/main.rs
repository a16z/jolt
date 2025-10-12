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

    let first_input: &[u8] = &[5u8; 32];
    let second_input = [6u8; 32];
    let third_input = [7u8; 32];

    let (trusted_advice_commitment, _hint) = guest::commit_trusted_advice_merkle_tree(
        TrustedAdvice::new(second_input),
        &prover_preprocessing,
    );

    let prove_merkle_tree = guest::build_prover_merkle_tree(program, prover_preprocessing.clone());
    let verify_merkle_tree = guest::build_verifier_merkle_tree(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_merkle_tree(
        first_input,
        TrustedAdvice::new(second_input),
        UntrustedAdvice::new(third_input),
        trusted_advice_commitment.clone(),
    );
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    // Pass only the first input and trusted_advice commitment to the verifier
    let is_valid = verify_merkle_tree(
        first_input,
        output,
        program_io.panic,
        trusted_advice_commitment.clone(),
        proof,
    );

    info!("output: {}", hex::encode(output));
    info!("valid: {is_valid}");
}
