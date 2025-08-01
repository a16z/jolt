pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_test(target_dir);

    let prover_preprocessing = guest::preprocess_prover_test(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_test(&prover_preprocessing);

    let prove_test = guest::build_prover_test(program, prover_preprocessing);
    let verify_test = guest::build_verifier_test(verifier_preprocessing);

    let (output, proof, program_io) = prove_test(50);
    let is_valid = verify_test(50, output, program_io.panic, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
