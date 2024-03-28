pub fn main() {
    let (prove_add, verify_add) = guest::build_add();
    let (prove_mul, verify_mul) = guest::build_mul();

    let (output, proof) = prove_add(5, 10);
    let is_valid = verify_add(proof);

    println!("add output: {}", output);
    println!("add valid: {}", is_valid);

    let (output, proof) = prove_mul(5, 10);
    let is_valid = verify_mul(proof);

    println!("mul output: {}", output);
    println!("mul valid: {}", is_valid);
}
