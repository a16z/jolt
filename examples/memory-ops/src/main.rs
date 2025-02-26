use std::time::Instant;

pub fn main() {
    let (prove, verify) = guest::build_memory_ops();

    let now = Instant::now();
    let (output, proof) = prove();
    println!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify(proof);

    println!(
        "outputs: {} {} {} {}",
        output.0, output.1, output.2, output.3
    );
    println!("valid: {}", is_valid);
}
