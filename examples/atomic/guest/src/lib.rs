#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;
use core::sync::atomic::{AtomicU64, Ordering};

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn atomic_test_u64(input: u64) -> (u64, u64) {
    // Test standard core::sync::atomic
    let atomic_std = AtomicU64::new(input);
    let loaded_std = atomic_std.load(Ordering::SeqCst);
    atomic_std.store(loaded_std + 1, Ordering::SeqCst);
    let incremented_std = atomic_std.fetch_add(1, Ordering::SeqCst);
    let result_std = black_box(incremented_std + 1);

    // Test portable-atomic (should work the same with passes=lower-atomic)
    let atomic_portable = portable_atomic::AtomicU64::new(input);
    let loaded_portable = atomic_portable.load(Ordering::SeqCst);
    atomic_portable.store(loaded_portable + 1, Ordering::SeqCst);
    let incremented_portable = atomic_portable.fetch_add(1, Ordering::SeqCst);
    let result_portable = black_box(incremented_portable + 1);

    // Return both results to verify they're identical
    (result_std, result_portable)
}
