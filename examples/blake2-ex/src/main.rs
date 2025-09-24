use std::time::Instant;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_blake2(target_dir);

    let prover_preprocessing = guest::preprocess_prover_blake2(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_blake2(&prover_preprocessing);

    let prove_blake2 = guest::build_prover_blake2(program, prover_preprocessing);
    let verify_blake2 = guest::build_verifier_blake2(verifier_preprocessing);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) = prove_blake2(input);
    tracing::info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_blake2(input, output, program_io.panic, proof);

    // Combine the two 32-byte arrays for display
    let (first_half, second_half) = output;
    let mut full_hash = [0u8; 64];
    full_hash[0..32].copy_from_slice(&first_half);
    full_hash[32..64].copy_from_slice(&second_half);

    tracing::info!("512-bit output: {}", hex::encode(full_hash));
    tracing::info!("valid: {is_valid}");
}
