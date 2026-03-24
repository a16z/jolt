use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    let target_dir = "/tmp/jolt-guest-targets";

    // int_to_string: max_trace_length = 65536
    info!("=== Int to String ===");
    let mut program = guest::compile_int_to_string(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_int_to_string(&mut program, chunk_count).unwrap();
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_int_to_string(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_int_to_string(&mut program).unwrap();
        let prover_preprocessing =
            guest::preprocess_prover_int_to_string(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_int_to_string(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove = guest::build_prover_int_to_string(program, prover_preprocessing);
    let verify = guest::build_verifier_int_to_string(verifier_preprocessing);
    let (output, proof, program_io) = prove(81);
    info!("int to string output: {output:?}");

    let is_valid = verify(81, output, program_io.panic, proof);
    info!("int to string valid: {is_valid}");
    assert!(is_valid, "int_to_string proof verification failed!");

    // string_concat: max_trace_length = 131072
    info!("=== String Concat ===");
    let mut program = guest::compile_string_concat(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_string_concat(&mut program, chunk_count).unwrap();
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_string_concat(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_string_concat(&mut program).unwrap();
        let prover_preprocessing =
            guest::preprocess_prover_string_concat(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_string_concat(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove = guest::build_prover_string_concat(program, prover_preprocessing);
    let verify = guest::build_verifier_string_concat(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(20);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    info!("string concat output: {output:?}");

    let is_valid = verify(20, output, program_io.panic, proof);
    info!("string concat valid: {is_valid}");
    assert!(is_valid, "string_concat proof verification failed!");

    // parallel_sum_of_squares: max_trace_length = 1048576
    // Tests ZeroOS + Jolt threading with rayon
    info!("=== Parallel Sum of Squares (rayon) ===");
    let mut program = guest::compile_parallel_sum_of_squares(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_parallel_sum_of_squares(&mut program, chunk_count).unwrap();
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_parallel_sum_of_squares(
                &prover_preprocessing,
            );
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing =
            guest::preprocess_shared_parallel_sum_of_squares(&mut program).unwrap();
        let prover_preprocessing =
            guest::preprocess_prover_parallel_sum_of_squares(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_parallel_sum_of_squares(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove = guest::build_prover_parallel_sum_of_squares(program, prover_preprocessing);
    let verify = guest::build_verifier_parallel_sum_of_squares(verifier_preprocessing);

    let n = 101u32;
    let now = Instant::now();
    let (output, proof, program_io) = prove(n);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    info!(
        "parallel_sum_of_squares({}) = {} (expected: 348551)",
        n, output
    );

    let is_valid = verify(n, output, program_io.panic, proof);
    info!("parallel_sum_of_squares valid: {is_valid}");

    assert_eq!(output, 348551, "parallel sum of squares mismatch!");
    assert!(
        is_valid,
        "parallel_sum_of_squares proof verification failed!"
    );

    info!("=== All stdlib tests PASSED! ===");
}
