use sha2::{Digest, Sha256};

fn compute_native(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        let mut hasher = Sha256::new();
        hasher.update(&hash);
        let res = &hasher.finalize();
        hash = Into::<[u8; 32]>::into(*res);
    }

    hash
}

pub fn main() {
    let (prove_sha2_chain, verify_sha2_chain) = guest::build_sha2_chain();

    let input = [5u8; 32];
    let iters = 100;
    let native_output = compute_native(input, iters);
    let (output, proof) = prove_sha2_chain(input, iters);
    let is_valid = verify_sha2_chain(proof);

    assert_eq!(output, native_output, "output mismatch");
    println!("output: {}", hex::encode(output));
    println!("native_output: {}", hex::encode(native_output));
    println!("valid: {}", is_valid);
}
