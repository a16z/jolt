use ark_bn254::{Fr, G1Projective};
use std::time::Instant;

pub fn main() {
    let input: &[u8] = &[5u8; 2048];
    let (program, preprocessing) = guest::preprocess_sha3::<Fr, G1Projective>();
    let start = Instant::now();
    let (output, _proof, _commitments) =
        guest::prove_sha3::<Fr, G1Projective>(program, preprocessing, input);
    let end = Instant::now();
    let duration = end.duration_since(start);
    println!("Prover time: {}ms", duration.as_millis());

    println!("output: {}", hex::encode(output));
}
