#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_keccak256::Keccak256;

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn sha3(input: &[u8]) -> [u8; 32] {
    Keccak256::digest(input)
}

/// Hashes two rate blocks that are 8-byte aligned in guest memory: `digest`
/// then feeds the fused absorb inline straight from the caller's buffer
/// instead of staging each block through a stack copy.
#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn sha3_aligned(blocks: [[u64; 17]; 2]) -> [u8; 32] {
    // SAFETY: `blocks` is `size_of_val` initialized bytes and `u8` has no alignment requirement.
    let bytes =
        unsafe { core::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), size_of_val(&blocks)) };
    Keccak256::digest(bytes)
}
