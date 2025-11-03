use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_modexp_chain(target_dir);

    let prover_preprocessing = guest::preprocess_prover_modexp_chain(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_modexp_chain(&prover_preprocessing);

    let prove_modexp_chain = guest::build_prover_modexp_chain(program, prover_preprocessing);
    let verify_modexp_chain = guest::build_verifier_modexp_chain(verifier_preprocessing);

    // Example inputs: 256-bit base, exponent, and modulus
    let base = [5u8; 32]; // 256-bit base
    let exponent = [3u8; 32]; // 256-bit exponent
    let modulus = [7u8; 32]; // 256-bit modulus
    let iters = 10;

    let native_output = guest::modexp_chain(base, exponent, modulus, iters);
    let now = Instant::now();
    let (output, proof, program_io) = prove_modexp_chain(base, exponent, modulus, iters);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_modexp_chain(
        base,
        exponent,
        modulus,
        iters,
        output,
        program_io.panic,
        proof,
    );

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {}", hex::encode(output));
    info!("native_output: {}", hex::encode(native_output));
    info!("valid: {is_valid}");
}
