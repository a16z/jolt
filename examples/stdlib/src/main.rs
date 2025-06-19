use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_int_to_string(target_dir);

    let prover_preprocessing = guest::preprocess_prover_int_to_string(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_int_to_string(&prover_preprocessing);

    let prove = guest::build_prover_int_to_string(program, prover_preprocessing);
    let verify = guest::build_verifier_int_to_string(verifier_preprocessing);

    let (output, proof) = prove(81);
    println!("int to string output: {output:?}");

    let is_valid = verify(81, output, proof);
    println!("int to string valid: {is_valid}");

    let mut program = guest::compile_string_concat(target_dir);

    let prover_preprocessing = guest::preprocess_prover_string_concat(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_string_concat(&prover_preprocessing);

    let prove = guest::build_prover_string_concat(program, prover_preprocessing);
    let verify = guest::build_verifier_string_concat(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof) = prove(20);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    println!("string concat output: {output:?}");

    let is_valid = verify(20, output, proof);
    println!("string concat valid: {is_valid}");
}
