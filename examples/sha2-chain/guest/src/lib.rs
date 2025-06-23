#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable]
fn sha2_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        hash = jolt::sha256::Sha256::digest(&hash);
    }
    hash
}
