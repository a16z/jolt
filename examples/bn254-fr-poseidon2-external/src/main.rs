use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fr_poseidon2_external(target_dir);

    let shared_preprocessing = guest::preprocess_shared_fr_poseidon2_external(&mut program);
    let prover_preprocessing =
        guest::preprocess_prover_fr_poseidon2_external(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_fr_poseidon2_external(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );

    let prove = guest::build_prover_fr_poseidon2_external(program, prover_preprocessing);
    let verify = guest::build_verifier_fr_poseidon2_external(verifier_preprocessing);

    // Input state (1, 2, 3) — each Fr fits in a single u64 limb.
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];

    let now = Instant::now();
    let (output, proof, program_io) = prove(s0, s1, s2);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(s0, s1, s2, output, program_io.panic, proof);

    info!("output[0]: {:?}", output[0]);
    info!("output[1]: {:?}", output[1]);
    info!("output[2]: {:?}", output[2]);
    info!("valid: {is_valid}");
}
