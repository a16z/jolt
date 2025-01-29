pub fn main() {
    let (prove, verify) = guest::build_memory_ops();

    let (output, proof) = prove();
    let is_valid = verify(proof);

    println!(
        "outputs: {} {} {} {}",
        output.0, output.1, output.2, output.3
    );
    println!("valid: {}", is_valid);
}
