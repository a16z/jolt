use rand::Rng;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha2_chain(target_dir);

    let shared_preprocessing = guest::preprocess_shared_sha2_chain(&mut program);
    let prover_preprocessing = guest::preprocess_prover_sha2_chain(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_sha2_chain(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let input = [5u8; 32];
    let iters = 1000;
    
    // Create trusted advice data (16384 bytes = 2048*8) with random values
    let mut rng = rand::thread_rng();
    let mut trusted_data = [0u8; 16384];
    rng.fill(&mut trusted_data[..]);
    
    // Create untrusted advice data (8192 bytes = 1024*8) with random values
    let mut untrusted_data = [0u8; 8192];
    rng.fill(&mut untrusted_data[..]);

    // Build the prover closure (this moves `prover_preprocessing`), so compute any data we still
    // need from it first (e.g. trusted advice commitment/hint).
    let (trusted_advice_commitment, trusted_advice_hint) = guest::commit_trusted_advice_sha2_chain(
        jolt_sdk::TrustedAdvice::new(trusted_data.as_slice()),
        &prover_preprocessing,
    );

    let prove_sha2_chain = guest::build_prover_sha2_chain(program, prover_preprocessing);
    let verify_sha2_chain = guest::build_verifier_sha2_chain(verifier_preprocessing);
    
    let native_output = guest::sha2_chain(
        input,
        iters,
        jolt_sdk::TrustedAdvice::new(trusted_data.as_slice()),
        jolt_sdk::UntrustedAdvice::new(untrusted_data.as_slice()),
    );

    let now = Instant::now();
    let (output, proof, program_io) = prove_sha2_chain(
        input,
        iters,
        jolt_sdk::TrustedAdvice::new(trusted_data.as_slice()),
        jolt_sdk::UntrustedAdvice::new(untrusted_data.as_slice()),
        trusted_advice_commitment.clone(),
        trusted_advice_hint,
    );
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha2_chain(
        input,
        iters,
        output,
        program_io.panic,
        trusted_advice_commitment,
        proof,
    );

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {}", hex::encode(output));
    info!("native_output: {}", hex::encode(native_output));
    info!("valid: {is_valid}");
}
