// This is a variant of examples/merkle-tree that adds:
// 1. `--save` flag to serialize proof artifacts to /tmp/ for the transpiler pipeline
// 2. Transcript feature flags (transcript-poseidon, etc.) in Cargo.toml
//
// The upstream merkle-tree example is left unmodified. This crate reuses its guest.

use jolt_sdk::{serialize_and_print_size, TrustedAdvice, UntrustedAdvice};
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_merkle_tree(target_dir);

    let shared_preprocessing = guest::preprocess_shared_merkle_tree(&mut program);
    let prover_preprocessing = guest::preprocess_prover_merkle_tree(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_merkle_tree(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let leaf1: &[u8] = &[5u8; 32];
    let leaf2 = [6u8; 32];
    let leaf3 = [7u8; 32];
    let leaf4 = [8u8; 32];

    let (trusted_advice_commitment, trusted_advice_hint) =
        guest::commit_trusted_advice_merkle_tree(
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

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/merkle_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/merkle_io_device.bin", &program_io)
            .expect("Could not serialize io_device.");
        serialize_and_print_size(
            "Trusted Advice Commitment",
            "/tmp/merkle_trusted_advice.bin",
            &trusted_advice_commitment,
        )
        .expect("Could not serialize trusted advice commitment.");
    }

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
