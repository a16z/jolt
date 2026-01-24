//! sig-recovery host program
//!
//! This is the host-side program that:
//! 1. Generates test transactions
//! 2. Compiles the guest program
//! 3. Proves the execution
//! 4. Verifies the proof

use sig_recovery::{generate_test_transactions, serialize_transactions};
use std::time::Instant;
use tracing::info;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("sig-recovery: zkVM ECDSA Signature Recovery");
    info!("=============================================\n");

    // Generate test transactions
    let tx_count = 5;
    info!("Generating {} test transactions...", tx_count);
    let start = Instant::now();
    let txs = generate_test_transactions(tx_count);
    info!("Generation time: {:?}", start.elapsed());

    // Serialize transactions for the guest
    info!("Serializing transactions...");
    let txs_bytes = serialize_transactions(&txs);
    info!("Serialized size: {} bytes", txs_bytes.len());

    // Compile the guest program
    let target_dir = "/tmp/jolt-guest-targets";
    info!("\nCompiling guest program...");
    let start = Instant::now();
    let mut program = guest::compile_verify_txs(target_dir);
    info!("Compile time: {:?}", start.elapsed());

    info!("\nPreprocessing...");
    let start = Instant::now();
    let prover_preprocessing = guest::preprocess_prover_verify_txs(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_verify_txs(&prover_preprocessing);
    info!("Preprocessing time: {:?}", start.elapsed());

    let prove_verify_txs = guest::build_prover_verify_txs(program, prover_preprocessing);
    let verify_verify_txs = guest::build_verifier_verify_txs(verifier_preprocessing);

    // Analyze trace length
    info!("\nAnalyzing trace...");
    let program_summary = guest::analyze_verify_txs(&txs_bytes);
    let trace_length = program_summary.trace.len();
    let max_trace_length = if trace_length == 0 {
        1
    } else {
        (trace_length - 1).next_power_of_two()
    };
    info!("Trace length: {}", trace_length);
    info!("Max trace length: {}", max_trace_length);
    drop(program_summary); // Free trace memory before proving

    // Prove
    info!("\nProving {} transactions...", tx_count);
    let start = Instant::now();
    let (output, proof, program_io) = prove_verify_txs(&txs_bytes);
    let prove_time = start.elapsed();
    info!("Prove time: {:?}", prove_time);

    // Check output
    info!("\nVerification Result:");
    info!("  Transactions processed: {}", output.tx_count);
    info!("  Signers recovered: {}", output.recovered_count);

    if !output.signers.is_empty() {
        info!("  First signer: 0x{}", hex::encode(output.signers[0]));
    }

    // Verify
    info!("\nVerifying proof...");
    let start = Instant::now();
    let is_valid = verify_verify_txs(&txs_bytes, output, program_io.panic, proof);
    let verify_time = start.elapsed();
    info!("Verify time: {:?}", verify_time);
    info!("Proof valid: {}", is_valid);

    if is_valid {
        info!("\n✅ sig-recovery zkVM transaction verification successful!");
    } else {
        info!("\n❌ Proof verification failed!");
    }
}
