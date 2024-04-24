pub fn main() {
    let (prove_sha2, verify_sha2) = guest::build_sha2();

    let input: &[u8] = &[5u8; 32];
    let (output, proof) = prove_sha2(input);
    let is_valid = verify_sha2(proof);

    let input = [5u8; 32];
    let iters = 100;
    let (program, preprocessing) = guest::preprocess_sha2_chain();
    let (output, proof) = guest::prove_sha2_chain(program, preprocessing, input, iters);
    let verify = proof.verify();

    println!("output: {}", hex::encode(output));
    println!("valid: {}", is_valid);
}
