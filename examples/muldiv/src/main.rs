pub fn main() {
    let (prove, verify) = guest::build_muldiv();

    let (output, proof) = prove(12031293, 17, 92);
    let is_valid = verify(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
