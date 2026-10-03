#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, max_trace_length = 65536)]
fn sha2(input: &[u8]) -> [u8; 32] {
    jolt_inlines_sha2::Sha256::digest(input)
}
