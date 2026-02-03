use std::time::Instant;
use tracing::info;

// Demonstration of advice tape usage in a provable computation
pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";

    // example input
    let n = 221u8;
    let a = vec![1usize, 2, 3, 4, 5];
    let b = vec![0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    let mut program = guest::compile_advice_demo(target_dir);
    let shared_preprocessing = guest::preprocess_shared_advice_demo(&mut program);
    let prover_preprocessing = guest::preprocess_prover_advice_demo(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_advice_demo(shared_preprocessing, verifier_setup);
    let prove_advice_demo = guest::build_prover_advice_demo(program, prover_preprocessing);
    let verify_advice_demo = guest::build_verifier_advice_demo(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_advice_demo(n, a.clone(), b.clone());
    let prove_time = now.elapsed();
    info!("Prover runtime: {} s", prove_time.as_secs_f64());

    let trace_length_advice = proof.trace_length;

    let is_valid_advice = verify_advice_demo(n, a, b, output, program_io.panic, proof);

    info!("Proof valid: {}", is_valid_advice);
    info!("Trace length: {} cycles", trace_length_advice);

    assert!(is_valid_advice, "Proof verification failed");
}
