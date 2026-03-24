use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_alloc(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_alloc(&mut program, chunk_count).unwrap();
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_alloc(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_alloc(&mut program).unwrap();
        let prover_preprocessing = guest::preprocess_prover_alloc(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_alloc(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

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
