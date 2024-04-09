pub fn main() {
    let (prove_sha3_chain, verify_sha3_chain) = guest::build_sha3_chain();

    let input = [5u8; 32];
    let iters = 100;
    let (output, proof) = prove_sha3_chain(input, iters);
    let is_valid = verify_sha3_chain(proof);

    println!("output: {}", hex::encode(output));
    println!("valid: {}", is_valid);
}
