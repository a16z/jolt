use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_int_to_string(target_dir);

    let shared_preprocessing = guest::preprocess_shared_int_to_string(&mut program);
    let prover_preprocessing = guest::preprocess_prover_int_to_string(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_int_to_string(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove = guest::build_prover_int_to_string(program, prover_preprocessing);
    let verify = guest::build_verifier_int_to_string(verifier_preprocessing);
    let (output, proof, program_io) = prove(81);
    info!("int to string output: {output:?}");

    let is_valid = verify(81, output, program_io.panic, proof);
    info!("int to string valid: {is_valid}");

    let mut program = guest::compile_string_concat(target_dir);

    let shared_preprocessing = guest::preprocess_shared_string_concat(&mut program);
    let prover_preprocessing = guest::preprocess_prover_string_concat(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_string_concat(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove = guest::build_prover_string_concat(program, prover_preprocessing);
    let verify = guest::build_verifier_string_concat(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(20);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    info!("string concat output: {output:?}");

    let is_valid = verify(20, output, program_io.panic, proof);
    info!("string concat valid: {is_valid}");
}
