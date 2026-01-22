#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn sha3(input: &[u8]) -> [u8; 32] {
    jolt_inlines_keccak256::Keccak256::digest(input)
}
