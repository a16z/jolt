//! The test binary's heap accounting, for tests that bound the bytes a
//! kernel step holds.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live and peak heap bytes across every thread of this test process,
/// with a reallocation counted as its new buffer beside the old one.
struct CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

impl CountingAllocator {
    fn add(bytes: usize) {
        let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let _ = PEAK.fetch_max(live, Ordering::Relaxed);
    }

    fn sub(bytes: usize) {
        let _ = LIVE.fetch_sub(bytes, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards its request unchanged to `System`, which
// upholds the `GlobalAlloc` contract; the counters only observe sizes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::add(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::add(layout.size());
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) };
        Self::sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::add(new_size);
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        Self::sub(if moved.is_null() {
            new_size
        } else {
            layout.size()
        });
        moved
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// `run`'s result and the most bytes it held at once beyond those live
/// when it started. Every thread of the Rayon pool runs once first: a pool
/// thread allocates as it starts, which a busy machine can delay past an
/// earlier call. The caller still runs its operation once before, so that
/// only `run`'s own requests move the counters.
pub(crate) fn peak_above_start<R>(run: impl FnOnce() -> R) -> (R, usize) {
    #[cfg(feature = "parallel")]
    let _ = rayon::broadcast(|_| ());
    let start = LIVE.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let result = run();
    (result, PEAK.load(Ordering::Relaxed) - start)
}
