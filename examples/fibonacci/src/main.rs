use jolt_sdk::Serializable;
use std::time::Instant;

pub fn main() {
    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let prover_preprocessing = guest::preprocess_prover_fib(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_fib(&prover_preprocessing);

    if save_to_disk {
        verifier_preprocessing
            .save_to_target_dir("/tmp")
            .expect("failed to save verifier preprocessing");
    }

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let program_summary = guest::analyze_fib(10);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let now = Instant::now();
    let (output, io_device, proof) = prove_fib(50);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    if save_to_disk {
        proof
            .save_to_file("/tmp/fib_proof.bin")
            .expect("failed to save proof");
        io_device
            .save_to_file("/tmp/fib_io_device.bin")
            .expect("failed to save io device");
    }

    let is_valid = verify_fib(50, output, proof);
    println!("output: {output}");
    println!("valid: {is_valid}");
}
