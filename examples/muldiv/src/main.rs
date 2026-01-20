use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_muldiv(target_dir);

    let prover_preprocessing = guest::preprocess_muldiv(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_muldiv(&prover_preprocessing);

    let prove = guest::build_prover_muldiv(program, prover_preprocessing);
    let verify = guest::build_verifier_muldiv(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(12031293, 17, 92);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(12031293, 17, 92, output, program_io.panic, proof);

    info!("output: {output}");
    info!("valid: {is_valid}");
}
