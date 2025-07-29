#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha2(input: &[u8]) -> [u8; 32] {
    // Use Jolt's optimized SHA256 implementation
    jolt::sha256::Sha256::digest(input)
}
