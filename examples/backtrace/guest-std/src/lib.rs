/// std backtrace demo: intentionally panic after a few stack frames.
/// Note: std mode requires a large trace length due to std library initialization.
#[jolt::provable(
    heap_size = 32768,
    stack_size = 1048576,
    max_trace_length = 524288,
    profile = "guest",
    backtrace = "dwarf"
)]
fn panic_backtrace_std(should_panic: bool) -> u32 {
    if should_panic {
        level_one();
    }
    11
}

fn level_one() {
    level_two();
}

fn level_two() {
    panic!("backtrace demo (std)");
}
