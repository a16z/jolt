#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 65536, max_trace_length = 65536)]
fn add(x: u32, y: u32) -> u32 {
    x + y
}

#[jolt::provable(memory_size = 65536, max_trace_length = 65536)]
fn mul(x: u32, y: u32) -> u32 {
    x * y
}
