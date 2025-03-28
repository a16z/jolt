use std::time::Instant;

use jolt_sdk::JoltHyperKZGProof;

pub fn main() {
    let (prove_add, verify_add) = build_add();
    let (prove_mul, verify_mul) = build_mul();

    let now = Instant::now();
    let (output, proof) = prove_add(5, 10);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_add(5, 10, output, proof);

    println!("add output: {}", output);
    println!("add valid: {}", is_valid);

    let (output, proof) = prove_mul(5, 10);
    let is_valid = verify_mul(5, 10, output, proof);

    println!("mul output: {}", output);
    println!("mul valid: {}", is_valid);
}

fn build_add() -> (
    impl Fn(u32, u32) -> (u32, JoltHyperKZGProof) + Sync + Send,
    impl Fn(u32, u32, u32, JoltHyperKZGProof) -> bool + Sync + Send,
) {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_add(target_dir);

    let prover_preprocessing = guest::preprocess_prover_add(&program);
    let verifier_preprocessing = guest::preprocess_verifier_add(&program);

    let prove = guest::build_prover_add(program, prover_preprocessing);
    let verify = guest::build_verifier_add(verifier_preprocessing);

    (prove, verify)
}

fn build_mul() -> (
    impl Fn(u32, u32) -> (u32, JoltHyperKZGProof) + Sync + Send,
    impl Fn(u32, u32, u32, JoltHyperKZGProof) -> bool + Sync + Send,
) {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_mul(target_dir);

    let prover_preprocessing = guest::preprocess_prover_mul(&program);
    let verifier_preprocessing = guest::preprocess_verifier_mul(&program);

    let prove = guest::build_prover_mul(program, prover_preprocessing);
    let verify = guest::build_verifier_mul(verifier_preprocessing);

    (prove, verify)
}
