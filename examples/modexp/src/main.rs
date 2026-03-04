use std::time::Instant;
use tracing::info;

const BIT_LENGTH: usize = 256;
const NUM_ITERS: u32 = 10;

fn random_biguint(bits: usize) -> num_bigint::BigUint {
    use rand::Rng;
    let bytes = bits.div_ceil(8);
    let mut buf = vec![0u8; bytes];
    rand::thread_rng().fill(&mut buf[..]);
    // Ensure the top bit is set so we get a full-width number
    buf[0] |= 0x80;
    num_bigint::BigUint::from_bytes_be(&buf)
}

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";

    let base = random_biguint(BIT_LENGTH);
    let exp = random_biguint(BIT_LENGTH);
    // Ensure modulus is odd (required for meaningful modexp)
    let mut modulus = random_biguint(BIT_LENGTH);
    modulus |= num_bigint::BigUint::from(1u8);

    let base_bytes = base.to_bytes_be();
    let exp_bytes = exp.to_bytes_be();
    let modulus_bytes = modulus.to_bytes_be();

    info!(
        "Modular exponentiation: {}-bit base^exp mod modulus, {} iterations",
        BIT_LENGTH, NUM_ITERS
    );

    let native_output = guest::modexp(
        base_bytes.clone(),
        exp_bytes.clone(),
        modulus_bytes.clone(),
        NUM_ITERS,
    );

    let mut program = guest::compile_modexp(target_dir);
    let shared_preprocessing = guest::preprocess_shared_modexp(&mut program);
    let prover_preprocessing = guest::preprocess_prover_modexp(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_modexp(shared_preprocessing, verifier_setup);
    let prove_modexp = guest::build_prover_modexp(program, prover_preprocessing);
    let verify_modexp = guest::build_verifier_modexp(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_modexp(
        base_bytes.clone(),
        exp_bytes.clone(),
        modulus_bytes.clone(),
        NUM_ITERS,
    );
    let prove_time = now.elapsed();
    info!("Prover runtime: {} s", prove_time.as_secs_f64());

    let is_valid = verify_modexp(
        base_bytes,
        exp_bytes,
        modulus_bytes,
        NUM_ITERS,
        output.clone(),
        program_io.panic,
        proof,
    );

    assert_eq!(output, native_output, "output mismatch");
    info!("Output: {} bytes", output.len());
    info!("Proof valid: {}", is_valid);
}
