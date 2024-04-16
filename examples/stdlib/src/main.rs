pub fn main() {
    let (prove, verify) = guest::build_fun_with_strings();

    let (output, proof) = prove(41);
    let is_valid = verify(proof);

    println!("output: {:?}", output);
    println!("valid: {}", is_valid);
}
