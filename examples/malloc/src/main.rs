use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_alloc(target_dir);

    let shared_preprocessing = guest::preprocess_shared_alloc(&mut program);
    let prover_preprocessing = guest::preprocess_prover_alloc(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_alloc(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove = guest::build_prover_alloc(program, prover_preprocessing);
    let verify = guest::build_verifier_alloc(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(12345);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(12345, output, program_io.panic, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
