use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_alloc(target_dir);

    let prover_preprocessing = guest::preprocess_prover_alloc(&program);
    let verifier_preprocessing = guest::preprocess_verifier_alloc(&program);

    let prove_alloc = guest::build_prover_alloc(program, prover_preprocessing);
    let verify_alloc = guest::build_verifier_alloc(verifier_preprocessing);

    let now = Instant::now();
    let input = 41;
    let (output, proof) = prove_alloc(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_alloc(input, output, proof);

    println!("output: {output:?}");
    println!("valid: {is_valid}");
}
