use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_alloc(target_dir);

    let prover_preprocessing = guest::preprocess_alloc(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_alloc(&prover_preprocessing);

    let prove = guest::build_prover_alloc(program, prover_preprocessing);
    let verify = guest::build_verifier_alloc(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(12345);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(12345, output, program_io.panic, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
