use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    // Prove addition.
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_add(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_add(&mut program, chunk_count).unwrap();
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_add(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_add(&mut program).unwrap();
        let prover_preprocessing = guest::preprocess_prover_add(shared_preprocessing.clone());
        let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
        let verifier_preprocessing =
            guest::preprocess_verifier_add(shared_preprocessing, verifier_setup, None);
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove_add = guest::build_prover_add(program, prover_preprocessing);
    let verify_add = guest::build_verifier_add(verifier_preprocessing);

    // Prove multiplication.
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_mul(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_mul(&mut program, chunk_count).unwrap();
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_mul(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_mul(&mut program).unwrap();
        let prover_preprocessing = guest::preprocess_prover_mul(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_mul(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove_mul = guest::build_prover_mul(program, prover_preprocessing);
    let verify_mul = guest::build_verifier_mul(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_add(5, 10);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_add(5, 10, output, program_io.panic, proof);

    info!("add output: {output}");
    info!("add valid: {is_valid}");

    let (output, proof, program_io) = prove_mul(5, 10);
    let is_valid = verify_mul(5, 10, output, program_io.panic, proof);

    info!("mul output: {output}");
    info!("mul valid: {is_valid}");
}
