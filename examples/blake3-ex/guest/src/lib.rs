#![cfg_attr(feature = "guest", no_std)]

use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn blake3(input: &[u8]) -> [u8; 32] {
    // Use Jolt's optimized Blake2b implementation
    start_cycle_tracking("blake3_digest");
    let hash = jolt_inlines_blake3::Blake3::digest(input);
    end_cycle_tracking("blake3_digest");
    hash
}
