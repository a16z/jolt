use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha2(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing = guest::preprocess_committed_sha2(&mut program, chunk_count);
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_sha2(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_sha2(&mut program);
        let prover_preprocessing = guest::preprocess_prover_sha2(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_sha2(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove_sha2 = guest::build_prover_sha2(program, prover_preprocessing);
    let verify_sha2 = guest::build_verifier_sha2(verifier_preprocessing);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) = prove_sha2(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha2(input, output, program_io.panic, proof);

    info!("output: {}", hex::encode(output));
    info!("valid: {is_valid}");
}
