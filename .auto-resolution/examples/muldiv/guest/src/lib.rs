#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[jolt::provable(memory_size = 32768, max_trace_length = 65536)]
fn muldiv(a: u32, b: u32, c: u32) -> u32 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("muldiv");
    let result = black_box(a * b / c); // use black_box to keep code in place
    end_cycle_tracking("muldiv");
    result
}
