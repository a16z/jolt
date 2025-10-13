use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_demo(target_dir);

    let prover_preprocessing = guest::preprocess_prover_demo(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_demo(&prover_preprocessing);

    let prove = guest::build_prover_demo(program, prover_preprocessing);
    let verify = guest::build_verifier_demo(verifier_preprocessing);

    let test_input = 42;

    let now = Instant::now();
    let (output, proof, program_io) = prove(test_input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify(test_input, output, program_io.panic, proof);

    info!("Output: {}", output);
    info!("Valid: {}", is_valid);
}
