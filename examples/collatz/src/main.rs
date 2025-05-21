use std::time::Instant;

pub fn main() {
    // Prove/verify convergence for a single number:
    let target_dir = "/tmp/jolt-guest-targets";
    let program = guest::compile_collatz_convergence(target_dir);

    let prover_preprocessing = guest::preprocess_prover_collatz_convergence(&program);
    let verifier_preprocessing = guest::preprocess_verifier_collatz_convergence(&program);

    let prove_collatz_single =
        guest::build_prover_collatz_convergence(program, prover_preprocessing);
    let verify_collatz_single = guest::build_verifier_collatz_convergence(verifier_preprocessing);

    let now = Instant::now();
    let input = 19;
    let (output, proof) = prove_collatz_single(input);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_collatz_single(input, output, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");

    // Prove/verify convergence for a range of numbers:
    let program = guest::compile_collatz_convergence_range(target_dir);

    let prover_preprocessing = guest::preprocess_prover_collatz_convergence_range(&program);
    let verifier_preprocessing = guest::preprocess_verifier_collatz_convergence_range(&program);

    let prove_collatz_convergence =
        guest::build_prover_collatz_convergence_range(program, prover_preprocessing);
    let verify_collatz_convergence =
        guest::build_verifier_collatz_convergence_range(verifier_preprocessing);

    // https://www.reddit.com/r/compsci/comments/gk9x6g/collatz_conjecture_news_recently_i_managed_to/
    let start: u128 = 1 << 68;
    let now = Instant::now();
    let (output, proof) = prove_collatz_convergence(start, start + 100);
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_collatz_convergence(start, start + 100, output, proof);

    println!("output: {output}");
    println!("valid: {is_valid}");
}
