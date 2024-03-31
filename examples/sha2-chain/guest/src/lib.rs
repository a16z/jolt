#![cfg_attr(feature = "guest", no_std)]
#![no_main]

use sha2::{Sha256, Digest};

#[jolt::provable]
fn sha2_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        let mut hasher = Sha256::new();
        hasher.update(input);
        let res = &hasher.finalize();
        hash = Into::<[u8; 32]>::into (*res);
    }

    hash
}

