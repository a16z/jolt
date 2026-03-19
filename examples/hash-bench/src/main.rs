use std::time::Instant;

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_hashbench(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing = guest::preprocess_committed_hashbench(&mut program, chunk_count);
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_hashbench(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_hashbench(&mut program);
        let prover_preprocessing = guest::preprocess_prover_hashbench(shared_preprocessing.clone());
        let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
        let verifier_preprocessing =
            guest::preprocess_verifier_hashbench(shared_preprocessing, verifier_setup, None);
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove_hashbench = guest::build_prover_hashbench(program, prover_preprocessing);
    let verify_hashbench = guest::build_verifier_hashbench(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_hashbench();
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_hashbench(output, program_io.panic, proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {is_valid}");
}
