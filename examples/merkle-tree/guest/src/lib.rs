#![cfg_attr(feature = "guest", no_std)]

use core::ops::Deref;

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn merkle_tree(first_input: &[u8], second_input: jolt::UntrustedAdvice<[u8; 32]>) -> [u8; 32] {
    // Compute hash of first input
    let hash1 = jolt_inlines_sha2::Sha256::digest(first_input);
    let hash2 = jolt_inlines_sha2::Sha256::digest(second_input.deref());

    // Concatenate both hashes
    let mut concatenated = [0u8; 64];
    concatenated[..32].copy_from_slice(&hash1);
    concatenated[32..].copy_from_slice(&hash2);

    // Hash the concatenated result and return
    jolt_inlines_sha2::Sha256::digest(&concatenated)
}
