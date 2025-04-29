use std::time::Instant;

pub fn main() {
    // Prove addition.
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_add(target_dir);

    let prover_preprocessing = guest::preprocess_prover_add(&program);
    let verifier_preprocessing = guest::preprocess_verifier_add(&program);

    let prove_add = guest::build_prover_add(program, prover_preprocessing);
    let verify_add = guest::build_verifier_add(verifier_preprocessing);

    // Prove multiplication.
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_mul(target_dir);

    let prover_preprocessing = guest::preprocess_prover_mul(&program);
    let verifier_preprocessing = guest::preprocess_verifier_mul(&program);

    let prove_mul = guest::build_prover_mul(program, prover_preprocessing);
    let verify_mul = guest::build_verifier_mul(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof) = prove_add(5, 10);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_add(5, 10, output, proof);

    println!("add output: {output}");
    println!("add valid: {is_valid}");

    let (output, proof) = prove_mul(5, 10);
    let is_valid = verify_mul(5, 10, output, proof);

    println!("mul output: {output}");
    println!("mul valid: {is_valid}");
}
