use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_alloc(target_dir);

    let prover_preprocessing = guest::preprocess_alloc(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_alloc(&prover_preprocessing);

    let prove_alloc = guest::build_prover_alloc(program, prover_preprocessing);
    let verify_alloc = guest::build_verifier_alloc(verifier_preprocessing);

    let now = Instant::now();
    let input = 41;
    let (output, proof, program_io) = prove_alloc(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_alloc(input, output, program_io.panic, proof);

    info!("output: {output:?}");
    info!("valid: {is_valid}");
}
