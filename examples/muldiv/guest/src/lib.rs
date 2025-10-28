#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn muldiv(x: u32, y: u32) -> u32 {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "remw {result}, {a}, {b}",
            result = out(reg) result,
            a = in(reg) x, // 0_u32,
            b = in(reg) y, // 4294967295_u32,
        );
    }
    result
}
