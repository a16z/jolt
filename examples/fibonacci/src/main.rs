use jolt_sdk::serialize_and_print_size;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let prover_preprocessing = guest::preprocess_prover_fib(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_fib(&prover_preprocessing);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let program_summary = guest::analyze_fib(10);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let trace_file = "/tmp/fib_trace.bin";
    guest::trace_fib_to_file(trace_file, 50);
    info!("Trace file written to: {trace_file}.");

    let now = Instant::now();
    let (output, proof, io_device) = prove_fib(50);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/fib_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/fib_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    // Verify Stage 1 only (R1CS constraints) - for Groth16 transpilation experiment
    info!("Running Stage 1-only verification...");
    use jolt_core::zkvm::stage1_only_verifier::{
        Stage1OnlyPreprocessing, Stage1OnlyProof, Stage1OnlyVerifier,
    };
    use jolt_core::poly::commitment::dory::DoryCommitmentScheme;

    let (stage1_proof, opening_claims, commitments, ram_K) =
        Stage1OnlyProof::from_full_proof::<DoryCommitmentScheme>(&proof);

    let stage1_preprocessing = Stage1OnlyPreprocessing::new(proof.trace_length);

    let stage1_verifier = Stage1OnlyVerifier::new::<DoryCommitmentScheme>(
        stage1_preprocessing,
        stage1_proof,
        opening_claims,
        &io_device,
        &commitments,
        ram_K,
    )
    .expect("Failed to create Stage 1 verifier");

    stage1_verifier
        .verify()
        .expect("Stage 1 verification failed");
    info!("Stage 1 verification PASSED");

    let is_valid = verify_fib(50, output, io_device.panic, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}
