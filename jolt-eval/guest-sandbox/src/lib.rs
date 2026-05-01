#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(heap_size = 32768, stack_size = 65536, max_trace_length = 1048576)]
fn sandbox(input: &[u8]) -> u32 {
    // Simple hash — the red-team agent patches this to explore
    // code paths that might break soundness.
    let mut h: u32 = 0;
    for &b in input {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    h
}
