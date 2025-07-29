#![cfg_attr(feature = "guest", no_std)]

use sha3::{Digest, Keccak256};

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha3(input: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(input);
    let result = hasher.finalize();
    Into::<[u8; 32]>::into(result)
}
