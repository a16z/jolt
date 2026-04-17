use std::time::Duration;

/// Cumulative process CPU time (user + system) across all threads.
///
/// On Unix, wraps `getrusage(RUSAGE_SELF)`. Call delta = `process_cpu_time()`
/// at span exit minus at entry, divide by wall duration to get effective
/// core count for that span. Returns `Duration::ZERO` on unsupported platforms
/// or on error.
#[cfg(unix)]
pub(crate) fn process_cpu_time() -> Duration {
    // SAFETY: `rusage` is POD; all-zero bits are a valid initial state
    // for `libc::rusage`. `getrusage` overwrites the struct on success.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `ru` is a valid mutable reference to a `libc::rusage` on the
    // stack; `getrusage` only writes to the pointed-to struct and does not
    // retain the pointer past the call.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &raw mut ru) };
    if rc != 0 {
        return Duration::ZERO;
    }
    let user = Duration::new(
        ru.ru_utime.tv_sec as u64,
        (ru.ru_utime.tv_usec as u32) * 1000,
    );
    let sys = Duration::new(
        ru.ru_stime.tv_sec as u64,
        (ru.ru_stime.tv_usec as u32) * 1000,
    );
    user + sys
}

#[cfg(not(unix))]
pub(crate) fn process_cpu_time() -> Duration {
    Duration::ZERO
}
