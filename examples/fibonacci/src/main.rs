pub fn main() {
    let (program, preprocessing) = guest::preprocess_fib();
    let (output, _proof, _commitments) = guest::prove_fib(program, preprocessing, 50);
    println!("output: {}", output);
}

