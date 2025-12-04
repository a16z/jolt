use jolt_sdk::{serialize_and_print_size, TrustedAdvice};
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib2(target_dir);

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

    let prove_fib = guest::build_prover_fib2(program, prover_preprocessing.clone());
    let verify_fib = guest::build_verifier_fib2(verifier_preprocessing);
    let advice_data = [53u8, 100, 112, 56, 57, 230, 59, 54, 73];
    let (trusted_advice_commitment, _hint) = guest::commit_trusted_advice_fib2(
        TrustedAdvice::new(advice_data),
        &prover_preprocessing,
    );

    // let program_summary = guest::analyze_fib(10);
    // program_summary
    //     .write_to_file("fib_10.txt".into())
    //     .expect("should write");

    let trace_file = "/tmp/fib_trace.bin";
    guest::trace_fib2_to_file(trace_file, 10, TrustedAdvice::new(advice_data));
    info!("Trace file written to: {trace_file}.");

    let now = Instant::now();
    let (output, proof, io_device) = prove_fib(10, TrustedAdvice::new(advice_data), trusted_advice_commitment);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/fib_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/fib_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    let is_valid = verify_fib(10, output, io_device.panic, trusted_advice_commitment, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}
