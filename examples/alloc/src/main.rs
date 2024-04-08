pub fn main() {
    let (prove_alloc, verify_alloc) = guest::build_alloc();

    let (output, proof) = prove_alloc(41);
    let is_valid = verify_alloc(proof);

    println!("output: {:?}", output);
    println!("valid: {}", is_valid);
}
