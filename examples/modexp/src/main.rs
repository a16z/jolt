use std::time::Instant;
use tracing::info;

const NUM_ITERS: u32 = 10;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";

    let mut base = [0u8; 32];
    let mut exp = [0u8; 32];
    let mut modulus = [0u8; 32];
    rand::Rng::fill(&mut rand::thread_rng(), &mut base[..]);
    rand::Rng::fill(&mut rand::thread_rng(), &mut exp[..]);
    rand::Rng::fill(&mut rand::thread_rng(), &mut modulus[..]);
    modulus[31] |= 0x01; // Ensure odd

    info!(
        "Modular exponentiation: 256-bit base^exp mod modulus, {} iterations",
        NUM_ITERS
    );

    let native_output = guest::modexp(base, exp, modulus, NUM_ITERS);

    let mut program = guest::compile_modexp(target_dir);
    let shared_preprocessing = guest::preprocess_shared_modexp(&mut program);
    let prover_preprocessing = guest::preprocess_prover_modexp(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_modexp(shared_preprocessing, verifier_setup);
    let prove_modexp = guest::build_prover_modexp(program, prover_preprocessing);
    let verify_modexp = guest::build_verifier_modexp(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_modexp(base, exp, modulus, NUM_ITERS);
    let prove_time = now.elapsed();
    info!("Prover runtime: {} s", prove_time.as_secs_f64());

    let is_valid = verify_modexp(
        base,
        exp,
        modulus,
        NUM_ITERS,
        output,
        program_io.panic,
        proof,
    );

    assert_eq!(output, native_output, "output mismatch");
    info!("Proof valid: {}", is_valid);
}
