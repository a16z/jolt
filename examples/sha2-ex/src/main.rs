pub fn main() {
    let input: &[u8] = &[5u8; 32];

    let (program, preprocessing) = guest::preprocess_sha2();
    let (output, _proof) = guest::prove_sha2(program, preprocessing, input);

    println!("output: {}", hex::encode(output));
}
