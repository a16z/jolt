#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha3(_input: &[u8]) -> [u8; 32] {
    todo!()
    // jolt::keccak256::Keccak256::digest(input)
}
