use jolt_sdk::{TrustedAdvice, UntrustedAdvice};
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_merkle_tree(target_dir);

    let shared_preprocessing = guest::preprocess_shared_merkle_tree(&mut program);
    let prover_preprocessing = guest::preprocess_prover_merkle_tree(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_merkle_tree(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let leaf1: &[u8] = &[5u8; 32];
    let leaf2 = [6u8; 32];
    let leaf3 = [7u8; 32];
    let leaf4 = [8u8; 32];

    let (trusted_advice_commitment, trusted_advice_hint) = guest::commit_trusted_advice_merkle_tree(
        TrustedAdvice::new(leaf2),
        TrustedAdvice::new(leaf3),
        &prover_preprocessing,
    );

    let prove_merkle_tree = guest::build_prover_merkle_tree(program, prover_preprocessing.clone());
    let verify_merkle_tree = guest::build_verifier_merkle_tree(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_merkle_tree(
        leaf1,
        TrustedAdvice::new(leaf2),
        TrustedAdvice::new(leaf3),
        UntrustedAdvice::new(leaf4),
        trusted_advice_commitment,
        trusted_advice_hint,
    );
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    // Pass only the first input and trusted_advice commitment to the verifier
    let is_valid = verify_merkle_tree(
        leaf1,
        output,
        program_io.panic,
        trusted_advice_commitment,
        proof,
    );

    info!("output: {}", hex::encode(output));
    info!("valid: {is_valid}");
}
