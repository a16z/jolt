#![cfg_attr(feature = "guest", no_std)]

use sha3::{Digest, Keccak256};

#[jolt::provable]
fn sha3_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        let mut hasher = Keccak256::new();
        hasher.update(input);
        let res = &hasher.finalize();
        hash = Into::<[u8; 32]>::into(*res);
    }

    hash
}
