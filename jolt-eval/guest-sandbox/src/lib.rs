#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, stack_size = 65536, max_trace_length = 1048576)]
fn sandbox(input: &[u8]) -> Vec<u8> {
    // Identity function — the red-team agent patches this to explore
    // code paths that might break soundness.
    input.to_vec()
}
