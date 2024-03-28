#![cfg_attr(feature = "guest", no_std)]
#![cfg_attr(feature = "guest", no_main)]

use sha2::{Sha256, Digest};

#[jolt_sdk::main]
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

