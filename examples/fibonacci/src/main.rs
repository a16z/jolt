pub fn main() {
    let (output, _proof) = guest::prove_fib(50);
    println!("output: {}", output);
}

