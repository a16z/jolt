#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use sha3::{Digest, Keccak256};

#[jolt::provable]
fn keccak1600_default(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    // Initialize message with hardcoded data using black_box to prevent optimization
    let mut message = black_box(*b"abcabcioqs1abca8wnwohxwmm881qsacsc11zfcscoqpxsscswdwewewqweiocsssqkk118csscsakchnlhoihwowhd1wiu120u3e12312bnjkbnkaqqqqqou9u092312111qww");

    let hash_result = [0u8; 32];
    // Perform iterations with black_box to prevent optimization
    for i in 0..black_box(num_iters) {
        let _iteration = black_box(i);

        // Compute digest of current message
        let hash_result = black_box(Keccak256::digest(&message));

        // Replace first byte of message with first byte of hash for next iteration
        message[0] = black_box(hash_result[0]);

        // Apply black_box to the modified message to prevent optimization
        message = black_box(message);
    }
    // Prevent final optimization of the result
    black_box(hash_result)
}
