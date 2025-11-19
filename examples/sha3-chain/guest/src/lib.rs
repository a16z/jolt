use jolt::jolt_println;

#[jolt::provable(memory_size = 32768, max_trace_length = 4194304)]
fn sha3_chain(input: [u8; 32], num_iters: u32) -> [u8; 32] {
    jolt_println!("sha3_chain called with num_iters: {}", num_iters);
    let mut hash = input;
    for _ in 0..num_iters {
        hash = jolt_inlines_keccak256::Keccak256::digest(&hash);
    }
    hash
}
