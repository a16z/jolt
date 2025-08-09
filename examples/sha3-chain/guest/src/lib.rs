#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn sha3_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let hash = input;
    for _ in 0..num_iters {
        todo!()
        // hash = jolt::keccak256::Keccak256::digest(&hash);
    }
    hash
}
