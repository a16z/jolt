use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_rand(target_dir);

    let prover_preprocessing = guest::preprocess_prover_rand(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_rand(&prover_preprocessing);

    let prove = guest::build_prover_rand(program, prover_preprocessing);
    let verify = guest::build_verifier_rand(verifier_preprocessing);
    let a = 1;
    let b = 10;

    let program_summary = guest::analyze_rand(a, b);
    let trace_length = program_summary.trace.len();
    let max_trace_length = if trace_length == 0 {
        1
    } else {
        (trace_length - 1).next_power_of_two()
    };
    println!("Trace length: {trace_length:?}");
    println!("Max trace length: {max_trace_length:?}");

    let now = Instant::now();
    let (output, proof, program_io) = prove(a, b);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(a, b, output, program_io.panic, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
