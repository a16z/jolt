use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";

    // Test case: compute the modular inverse of 3 modulo 11
    // Expected: 3 * 4 ≡ 12 ≡ 1 (mod 11), so the inverse is 4
    let a = 3u64;
    let m = 11u64;

    info!("Computing modular inverse of {} modulo {}", a, m);

    // Compile and preprocess the advice-based version
    let mut program = guest::compile_modinv(target_dir);
    let shared_preprocessing = guest::preprocess_shared_modinv(&mut program);
    let prover_preprocessing = guest::preprocess_prover_modinv(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_modinv(shared_preprocessing, verifier_setup);
    let prove_modinv = guest::build_prover_modinv(program, prover_preprocessing);
    let verify_modinv = guest::build_verifier_modinv(verifier_preprocessing);

    let now = Instant::now();
    let (output_advice, proof_advice, io_device_advice) = prove_modinv(a, m);
    let prove_time_advice = now.elapsed();
    info!("Prover runtime: {} s", prove_time_advice.as_secs_f64());

    let trace_length_advice = proof_advice.trace_length;

    let is_valid_advice = verify_modinv(a, m, output_advice, io_device_advice.panic, proof_advice);

    info!("Output (modular inverse): {}", output_advice);
    info!("Proof valid: {}", is_valid_advice);
    info!("Trace length: {} cycles", trace_length_advice);

    assert_eq!(output_advice, 4, "Expected inverse of 3 mod 11 to be 4");
    assert_eq!(((a as u128) * (output_advice as u128)) % (m as u128), 1);
    assert!(is_valid_advice, "Proof verification failed");
}
