use std::time::Instant;

pub fn main() {
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_fib(target_dir);

    let prover_preprocessing = guest::preprocess_prover_fib(&program);
    let verifier_preprocessing = guest::preprocess_verifier_fib(&program);

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let verify_fib = guest::build_verifier_fib(verifier_preprocessing);

    let program_summary = guest::analyze_fib(10);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_fib(50);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_fib(50, output, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
