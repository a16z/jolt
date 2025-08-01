#![cfg_attr(feature = "guest", no_std)]
#![allow(arithmetic_overflow)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn test(x: u32) -> u32 {
    3 % x.pow(x)
}
