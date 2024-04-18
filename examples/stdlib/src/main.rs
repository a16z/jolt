pub fn main() {
    let (prove, verify) = guest::build_int_to_string();

    let (output, proof) = prove(81);
    let is_valid = verify(proof);

    println!("output: {:?}", output);
    println!("valid: {}", is_valid);
}
