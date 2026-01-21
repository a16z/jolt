#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 32768, max_trace_length = 4194304)]
fn sha2_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        hash = jolt_inlines_sha2::Sha256::digest(&hash);
    }
    hash
}
