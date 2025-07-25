#![cfg_attr(feature = "guest", no_std)]

use blake2::{Blake2b512, Digest};
use core::hint::black_box;

#[jolt::provable]
fn blake2_default(input: [u8; 32], num_iters: u32) -> [u64; 8] {
    // Initialize message with hardcoded data using black_box to prevent optimization
    let mut message = black_box(*b"abcabcabcabccabkshfswisjsjfkisiwwwqqq88wmm88scsc11azfiocssqkk118csscsakchnlhoihwowhd1wiu120u3e12312bnjkbnkaqqqqqou9u092312111qww");

    let hash_result = [0u8; 10];
    // Perform iterations with black_box to prevent optimization
    for i in 0..black_box(num_iters) {
        let _iteration = black_box(i);

        // Compute digest of current message
        let hash_result = black_box(Blake2b512::digest(&message));

        // Replace first byte of message with first byte of hash for next iteration
        message[0] = black_box(hash_result[0]);

        // Apply black_box to the modified message to prevent optimization
        message = black_box(message);
    }

    // Compute final hash of the message
    let final_hash = black_box(Blake2b512::digest(&message));

    // Convert final hash (64 bytes) to [u64; 8] format for return
    let mut result = [0u64; 8];
    let hash_bytes = black_box(final_hash.as_slice());

    for i in 0..8 {
        result[i] = black_box(u64::from_le_bytes(
            hash_bytes[i * 8..(i + 1) * 8].try_into().unwrap(),
        ));
    }

    // Prevent final optimization of the result
    black_box(result)
}
