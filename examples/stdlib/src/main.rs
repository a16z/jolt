use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_int_to_string(target_dir);

    let prover_preprocessing = guest::preprocess_prover_int_to_string(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_int_to_string(&prover_preprocessing);

    let prove = guest::build_prover_int_to_string(program, prover_preprocessing);
    let verify = guest::build_verifier_int_to_string(verifier_preprocessing);
    let (output, proof, program_io) = prove(81);
    info!("int to string output: {output:?}");

    let is_valid = verify(81, output, program_io.panic, proof);
    info!("int to string valid: {is_valid}");

    // let mut program = guest::compile_string_concat(target_dir);

    // let prover_preprocessing = guest::preprocess_prover_string_concat(&mut program);
    // let verifier_preprocessing =
    //     guest::verifier_preprocessing_from_prover_string_concat(&prover_preprocessing);

    // let prove = guest::build_prover_string_concat(program, prover_preprocessing);
    // let verify = guest::build_verifier_string_concat(verifier_preprocessing);

    // let now = Instant::now();
    // let (output, proof, program_io) = prove(20);
    // info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    // info!("string concat output: {output:?}");

    // let is_valid = verify(20, output, program_io.panic, proof);
    // info!("string concat valid: {is_valid}");

    // Parallel sum of squares using rayon - tests ZeroOS + Jolt threading
    // Same as ZeroOS std-smoke test: expected result for n=101 is 348551
    // info!("=== Parallel Sum of Squares (rayon) ===");
    // let mut program = guest::compile_parallel_sum_of_squares(target_dir);

    // let prover_preprocessing = guest::preprocess_prover_parallel_sum_of_squares(&mut program);
    // let verifier_preprocessing =
    //     guest::verifier_preprocessing_from_prover_parallel_sum_of_squares(&prover_preprocessing);

    // let prove = guest::build_prover_parallel_sum_of_squares(program, prover_preprocessing);
    // let verify = guest::build_verifier_parallel_sum_of_squares(verifier_preprocessing);

    // let n = 101u32;
    // let now = Instant::now();
    // let (output, proof, program_io) = prove(n);
    // info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    // info!("parallel_sum_of_squares({}) = {} (expected: 348551)", n, output);

    // let is_valid = verify(n, output, program_io.panic, proof);
    // info!("parallel_sum_of_squares valid: {is_valid}");
    
    // assert_eq!(output, 348551, "parallel sum of squares mismatch!");
    // assert!(is_valid, "proof verification failed!");
    
    // info!("=== ZeroOS + Jolt rayon test PASSED! ===");
}
