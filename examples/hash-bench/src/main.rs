use std::time::Instant;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_hashbench(target_dir);

    let prover_preprocessing = guest::preprocess_hashbench(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_hashbench(&prover_preprocessing);

    let prove_hashbench = guest::build_prover_hashbench(program, prover_preprocessing);
    let verify_hashbench = guest::build_verifier_hashbench(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove_hashbench();
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_hashbench(output, program_io.panic, proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {is_valid}");
}
