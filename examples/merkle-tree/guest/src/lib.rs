#![cfg_attr(feature = "guest", no_std)]

use core::ops::Deref;

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn merkle_tree(
    first_input: &[u8],
    second_input: jolt::TrustedAdvice<[u8; 32]>,
    third_input: jolt::UntrustedAdvice<[u8; 32]>,
) -> [u8; 32] {
    // Compute hash of first input
    let hash1 = jolt_inlines_sha2::Sha256::digest(first_input);
    let hash2 = jolt_inlines_sha2::Sha256::digest(second_input.deref());
    let hash3 = jolt_inlines_sha2::Sha256::digest(third_input.deref());

    // Concatenate all three hashes
    let mut concatenated = [0u8; 96];
    concatenated[..32].copy_from_slice(&hash1);
    concatenated[32..64].copy_from_slice(&hash2);
    concatenated[64..].copy_from_slice(&hash3);

    // Hash the concatenated result and return
    jolt_inlines_sha2::Sha256::digest(&concatenated)
}
