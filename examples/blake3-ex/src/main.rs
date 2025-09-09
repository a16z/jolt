use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_blake3(target_dir);

    let prover_preprocessing = guest::preprocess_prover_blake3(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_blake3(&prover_preprocessing);

    let prove_blake3 = guest::build_prover_blake3(program, prover_preprocessing);
    let verify_blake3 = guest::build_verifier_blake3(verifier_preprocessing);

    let input: &[u8] = &[5u8; 32];
    let now = Instant::now();
    let (output, proof, program_io) = prove_blake3(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_blake3(input, output, program_io.panic, proof);

    let first_half = output;

    println!("output: {}", hex::encode(first_half));
    println!("valid: {is_valid}");
}
