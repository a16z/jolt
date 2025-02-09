use std::time::Instant;

pub fn main() {
    let (prove_fib, verify_fib) = guest::build_fib();
    let program_summary = guest::analyze_fib(10);
    program_summary
        .write_to_file("fib_10.txt".into())
        .expect("should write");

    let now = Instant::now();
    let (output, proof) = prove_fib(50);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_fib(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
