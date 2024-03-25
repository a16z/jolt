pub fn main() {
    let (prove_sha3, verify_sha3) = guest::build_sha3();

    let input: &[u8] = &[5u8; 32];
    let (output, proof) = prove_sha3(input);
    let is_valid = verify_sha3(proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {}", is_valid);
}
