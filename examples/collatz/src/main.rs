pub fn main() {
    // Prove/verify convergence for a single number:
    let (prove_collatz_single, verify_collatz_single) = guest::build_collatz_convergence();

    let (output, proof) = prove_collatz_single(19);
    let is_valid = verify_collatz_single(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);

    // Prove/verify convergence for a range of numbers:
    let (prove_collatz_convergence, verify_collatz_convergence) =
        guest::build_collatz_convergence_range();

    // https://www.reddit.com/r/compsci/comments/gk9x6g/collatz_conjecture_news_recently_i_managed_to/
    let start: u128 = 1 << 68;
    let (output, proof) = prove_collatz_convergence(start, start + 100);
    let is_valid = verify_collatz_convergence(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
