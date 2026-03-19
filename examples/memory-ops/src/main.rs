use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_memory_ops(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_memory_ops(&mut program, chunk_count);
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_memory_ops(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_memory_ops(&mut program);
        let prover_preprocessing =
            guest::preprocess_prover_memory_ops(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_memory_ops(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove = guest::build_prover_memory_ops(program, prover_preprocessing);
    let verify = guest::build_verifier_memory_ops(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove();
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(output, program_io.panic, proof);

    info!(
        "outputs: {} {} {} {}",
        output.0, output.1, output.2, output.3
    );
    info!("valid: {is_valid}");
}
