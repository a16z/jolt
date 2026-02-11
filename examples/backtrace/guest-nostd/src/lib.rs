#![cfg_attr(feature = "guest", no_std)]

/// no-std backtrace demo: intentionally panic after a few stack frames.
/// Note: Functions use stack arrays to prevent tail-call optimization,
/// ensuring proper frame pointers are generated for unwinding.
#[jolt::provable(
    heap_size = 32768,
    max_trace_length = 65536,
    stack_size = 1048576,
    profile = "guest",
    backtrace = "dwarf"
)]
fn panic_backtrace_nostd(should_panic: bool) -> u32 {
    if should_panic {
        level_one();
    }
    7
}

#[inline(never)]
fn level_one() {
    // Stack allocation prevents tail-call optimization
    let arr = [0u8; 32];
    core::hint::black_box(&arr);
    level_two();
}

#[inline(never)]
fn level_two() {
    // Stack allocation prevents tail-call optimization
    let arr = [0u8; 32];
    core::hint::black_box(&arr);
    panic!("backtrace demo (no-std)");
}
