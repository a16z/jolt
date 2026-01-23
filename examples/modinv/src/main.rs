use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_modinv(target_dir);

    let shared_preprocessing = guest::preprocess_shared_modinv(&mut program);

    let prover_preprocessing = guest::preprocess_prover_modinv(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_modinv(shared_preprocessing, verifier_setup);

    let prove_modinv = guest::build_prover_modinv(program, prover_preprocessing);
    let verify_modinv = guest::build_verifier_modinv(verifier_preprocessing);

    // Test case: compute the modular inverse of 3 modulo 11
    // Expected: 3 * 4 ≡ 12 ≡ 1 (mod 11), so the inverse is 4
    let a = 3u64;
    let m = 11u64;

    info!("Computing modular inverse of {} modulo {}", a, m);

    let now = Instant::now();
    let (output, proof, io_device) = prove_modinv(a, m);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify_modinv(a, m, output, io_device.panic, proof);

    info!("Input: a = {}, m = {}", a, m);
    info!("Output (modular inverse): {}", output);
    info!("Verification: a * output mod m = {} * {} mod {} = {}",
          a, output, m, ((a as u128) * (output as u128)) % (m as u128));
    info!("Proof valid: {}", is_valid);

    assert_eq!(output, 4, "Expected inverse of 3 mod 11 to be 4");
    assert_eq!(((a as u128) * (output as u128)) % (m as u128), 1);
    assert!(is_valid, "Proof verification failed");
}
