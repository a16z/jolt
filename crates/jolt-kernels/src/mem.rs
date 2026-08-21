//! Return freed-but-retained allocator pages to the OS.
//!
//! Stage-boundary frees mostly stay resident: macOS libmalloc keeps freed
//! MALLOC_LARGE regions in a fixed-depth "death row" ring (64 entries of up
//! to 512 MiB each on ≥32 GiB machines — at 2^25 cycles ~16 GiB of the
//! 31.7 GiB prover peak is such freed-but-retained pages), and glibc
//! similarly retains arena memory. A cheap explicit purge at chosen stage
//! boundaries returns those pages without touching mid-stage allocation
//! behavior.

#[cfg(target_os = "macos")]
extern "C" {
    /// `malloc/malloc.h`, stable since macOS 10.7. A NULL zone means all
    /// zones; `goal` 0 means release as much as possible. Returns bytes
    /// released (0 for the large cache — see [`pump_large_cache`]).
    fn malloc_zone_pressure_relief(zone: *mut core::ffi::c_void, goal: usize) -> usize;
}

/// Evict libmalloc's large-entry "death row" cache (macOS).
///
/// WHY a pump: on current macOS, `malloc_zone_pressure_relief` madvises the
/// tiny/small/medium racks but does NOT flush the large cache (verified
/// empirically on macOS 26: a 2 GiB freed hoard survives relief untouched).
/// The cache is a fixed-depth FIFO ring of freed regions; each new `free` of
/// a large block evicts — and thus `vm_deallocate`s — the oldest entry once
/// the ring is full. So: allocate `PUMP_SLOTS` never-touched large blocks and
/// free them all. The frees push every retained region out of the ring; the
/// pump blocks themselves are never faulted in, so they cost no RSS, no
/// zeroing, and leave only clean reservations behind.
///
/// Sizing: the ring holds ≤64 entries (`LARGE_ENTRY_CACHE_SIZE_HIGH`);
/// 128 slots covers it twice over. 9 MiB sits just above the medium/large
/// boundary (8 MiB on machines with the medium rack engaged), and the cache
/// only satisfies a request from an entry smaller than 2× the request — so
/// pump allocations cannot be served from (and thereby recycle) any retained
/// region ≥18 MiB, which is where the prover's multi-GiB retention lives.
/// If a future libmalloc moves the large threshold above 9 MiB the pump
/// degrades to a harmless no-op.
#[cfg(target_os = "macos")]
fn pump_large_cache() {
    const PUMP_SLOTS: usize = 128;
    const PUMP_BYTES: usize = 9 << 20;

    let mut slots = [core::ptr::null_mut::<core::ffi::c_void>(); PUMP_SLOTS];
    for slot in &mut slots {
        // SAFETY: plain malloc; the pointer is never dereferenced (the pages
        // must stay untouched) and is freed below.
        *slot = unsafe { libc::malloc(PUMP_BYTES) };
    }
    for slot in slots {
        // SAFETY: `slot` came from `malloc` above (or is NULL, which `free`
        // accepts).
        unsafe { libc::free(slot) };
    }
}

/// Cycle domains at least this large purge freed staging pages at kernels'
/// internal lifetime boundaries (see [`purge_staging`]); below it the
/// retained pages are too small to matter against the purge's fixed cost.
pub(crate) const PURGE_MIN_LOG_T: usize = 22;

/// Return freed-but-retained allocator pages mid-stage. Stage boundaries
/// already purge (the prover driver), but several kernels free multi-GiB
/// staging generations *inside* a stage at their representation-transition
/// boundaries, and those pages otherwise sit in the allocator's
/// freed-large-block cache until the next stage boundary, inflating the
/// stage's resident high-water. Allocator-only: no value that reaches the
/// transcript is touched.
pub(crate) fn purge_staging(log_t: usize) {
    if log_t >= PURGE_MIN_LOG_T {
        let _ = release_retained_memory();
    }
}

/// Ask the allocator to hand freed-but-retained pages back to the OS.
///
/// Returns the number of bytes the platform reports released (macOS
/// pressure relief only — the large-cache pump and glibc's `malloc_trim`
/// don't report byte counts; per-stage RSS telemetry shows the real effect).
pub fn release_retained_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: FFI call with NULL zone (= all zones) and goal 0; takes no
        // pointers into our memory and only shrinks the allocator's caches.
        let relieved = unsafe { malloc_zone_pressure_relief(core::ptr::null_mut(), 0) };
        pump_large_cache();
        tracing::debug!(bytes_relieved = relieved, "released retained memory");
        relieved
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: no preconditions; trims the main arena's retained memory.
        let released = unsafe { libc::malloc_trim(0) };
        tracing::debug!(released, "malloc_trim");
        0
    }
    #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
    {
        0
    }
}
