#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn muldiv(a: i64, b: i64, c: u32) -> i64 {
    use jolt::{end_cycle_tracking, start_cycle_tracking};

    start_cycle_tracking("muldiv");
    let result = black_box(a / b); // use black_box to keep code in place
    end_cycle_tracking("muldiv");
    result
}
