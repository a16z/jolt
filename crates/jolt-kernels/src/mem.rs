//! Return allocator-retained pages to the OS at memory lifetime boundaries.

#[cfg(target_os = "macos")]
extern "C" {
    /// NULL zone targets all zones; zero goal releases as much as possible.
    fn malloc_zone_pressure_relief(zone: *mut core::ffi::c_void, goal: usize) -> usize;
}

/// Evict macOS libmalloc's 64-slot large-allocation cache. Untouched 9 MiB
/// blocks enter the cache without adding RSS and displace retained prover
/// regions; 128 slots covers the ring twice.
#[cfg(target_os = "macos")]
fn pump_large_cache() {
    const PUMP_SLOTS: usize = 128;
    const PUMP_BYTES: usize = 9 << 20;

    let mut slots = [core::ptr::null_mut::<core::ffi::c_void>(); PUMP_SLOTS];
    for slot in &mut slots {
        // SAFETY: the allocation is never dereferenced and is freed below.
        *slot = unsafe { libc::malloc(PUMP_BYTES) };
    }
    for slot in slots {
        // SAFETY: `slot` came from `malloc`; `free` also accepts NULL.
        unsafe { libc::free(slot) };
    }
}

/// Minimum cycle-domain size for mid-stage purges.
pub(crate) const PURGE_MIN_LOG_T: usize = 22;

/// Purge after a large staging generation dies.
pub(crate) fn purge_staging(log_t: usize) {
    if log_t >= PURGE_MIN_LOG_T {
        let _ = release_retained_memory();
    }
}

/// Return allocator-retained pages to the OS. Only macOS pressure relief
/// reports a byte count.
pub fn release_retained_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: NULL zone and zero goal are valid; no Rust pointer is passed.
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
