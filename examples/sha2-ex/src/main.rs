pub fn main() {
    let input: &[u8] = &[5u8; 32];
    let (output, _proof) = guest::prove_sha2(input);

    println!("output: {}", hex::encode(output));
}
