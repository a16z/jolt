use std::time::Instant;

pub fn main() {
    let (program, preprocessing) = guest::preprocess_sha3();

    let input: &[u8] = &[5u8; 32];

    let start = Instant::now();

    let (output, _proof) = guest::prove_sha3(program, preprocessing, input);

    let end = Instant::now();
    let duration = end.duration_since(start);

    println!("prover time: {}ms", duration.as_millis());
    println!("output: {}", hex::encode(output));
}
