#![cfg_attr(feature = "guest", no_std)]

use core::ops::Deref;

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha2(public_input: &[u8], second_input: jolt::Private<[u8; 32]>) -> [u8; 32] {
    // Compute hash of first input
    let hash1 = jolt_inlines_sha2::Sha256::digest(public_input);

    // Compute hash of second input
    let hash2 = jolt_inlines_sha2::Sha256::digest(second_input.deref());

    // Concatenate both hashes
    let mut concatenated = [0u8; 64];
    concatenated[..32].copy_from_slice(&hash1);
    concatenated[32..].copy_from_slice(&hash2);

    // Hash the concatenated result and return
    jolt_inlines_sha2::Sha256::digest(&concatenated)
}

// _private_input: jolt::Private<[u8; 32]>

// ee6e5d895a0c2818f0f4e45bdabae0af3bc6c1535551f6b180bb43c420be91bc
