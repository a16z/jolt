//! Memory profiling utilities.
//!
//! Tracks physical memory deltas across labeled spans. Call
//! [`start_memory_tracing_span`] before the section and
//! [`end_memory_tracing_span`] after, then [`report_memory_usage`] to
//! log all collected deltas.

use memory_stats::memory_stats;
use std::{
    collections::BTreeMap,
    sync::{LazyLock, Mutex},
};

use crate::units::{format_memory_size, BYTES_PER_GIB};

static MEMORY_USAGE_MAP: LazyLock<Mutex<BTreeMap<&'static str, f64>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
static MEMORY_DELTA_MAP: LazyLock<Mutex<BTreeMap<&'static str, f64>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Records the current physical memory usage at the start of a labeled span.
///
/// Logs a warning and returns without recording if memory stats are unavailable
/// or if a span with the same label is already open.
pub fn start_memory_tracing_span(label: &'static str) {
    let Some(stats) = memory_stats() else {
        tracing::warn!(
            span = label,
            "memory stats unavailable, skipping span start"
        );
        return;
    };
    let memory_gib = stats.physical_mem as f64 / BYTES_PER_GIB;
    let mut map = MEMORY_USAGE_MAP.lock().unwrap_or_else(|e| e.into_inner());
    if map.insert(label, memory_gib).is_some() {
        tracing::warn!(span = label, "duplicate memory span label, overwriting");
    }
}

/// Closes a labeled memory span and records the memory delta (in GiB).
///
/// Logs a warning and returns without recording if memory stats are unavailable
/// or if no matching span was opened.
pub fn end_memory_tracing_span(label: &'static str) {
    let Some(stats) = memory_stats() else {
        tracing::warn!(span = label, "memory stats unavailable, skipping span end");
        return;
    };
    let memory_gib_end = stats.physical_mem as f64 / BYTES_PER_GIB;
    let Some(memory_gib_start) = MEMORY_USAGE_MAP
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(label)
    else {
        tracing::warn!(span = label, "no open memory span, skipping span end");
        return;
    };

    let delta = memory_gib_end - memory_gib_start;
    let _ = MEMORY_DELTA_MAP
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(label, delta);
}

/// Logs all collected memory deltas and warns about any unclosed spans.
pub fn report_memory_usage() {
    let memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap_or_else(|e| e.into_inner());
    for label in memory_usage_map.keys() {
        tracing::warn!(span = label, "unclosed memory tracing span");
    }

    let memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap_or_else(|e| e.into_inner());
    for (label, delta) in memory_delta_map.iter() {
        tracing::info!(
            span = label,
            delta = %format_memory_size(*delta),
            "memory delta"
        );
    }
}

/// Process-lifetime peak RSS in bytes, from `getrusage(RUSAGE_SELF)`.
///
/// The kernel-maintained high-water mark — unlike a sampling monitor it
/// cannot miss short allocation spikes, so it is the headline memory number
/// for benchmark runs. `ru_maxrss` is reported in bytes on macOS and
/// kibibytes on Linux; normalized to bytes here. Returns `None` on
/// non-unix targets or if the syscall fails.
#[cfg(unix)]
pub fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: getrusage writes a complete rusage struct into the provided
    // storage on success (return value 0).
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &raw mut usage) != 0 {
            return None;
        }
        usage
    };
    let raw = usage.ru_maxrss as u64;
    Some(if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    })
}

/// Process-lifetime peak RSS in bytes. Always `None` on non-unix targets.
#[cfg(not(unix))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

/// Kernel physical-footprint counters (`phys_footprint`, what `footprint(1)`
/// and `time -l` report): dirty host memory plus Metal allocations,
/// including Private buffers that never enter RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FootprintSample {
    pub current_bytes: u64,
    /// High-water mark since the last [`reset_footprint_interval`].
    pub interval_peak_bytes: u64,
    pub lifetime_peak_bytes: u64,
}

#[cfg(target_os = "macos")]
extern "C" {
    /// Exported by libsystem_kernel (private libproc header): restarts the
    /// interval high-water mark reported as `ri_interval_max_phys_footprint`.
    fn proc_reset_footprint_interval(pid: libc::c_int) -> libc::c_int;
}

/// This process's footprint counters; `None` off macOS or if the call fails.
#[cfg(target_os = "macos")]
pub fn phys_footprint() -> Option<FootprintSample> {
    use libc::{rusage_info_t, rusage_info_v4, RUSAGE_INFO_V4};
    // SAFETY: proc_pid_rusage writes one complete rusage_info_v4 into the
    // provided storage for flavor RUSAGE_INFO_V4 and returns 0 on success.
    let info = unsafe {
        let mut info: rusage_info_v4 = std::mem::zeroed();
        if libc::proc_pid_rusage(
            libc::getpid(),
            RUSAGE_INFO_V4,
            (&raw mut info).cast::<rusage_info_t>(),
        ) != 0
        {
            return None;
        }
        info
    };
    Some(FootprintSample {
        current_bytes: info.ri_phys_footprint,
        interval_peak_bytes: info.ri_interval_max_phys_footprint,
        lifetime_peak_bytes: info.ri_lifetime_max_phys_footprint,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn phys_footprint() -> Option<FootprintSample> {
    None
}

/// Restarts the interval high-water mark read by [`phys_footprint`].
pub fn reset_footprint_interval() {
    #[cfg(target_os = "macos")]
    // SAFETY: the call takes only a pid and writes no caller memory.
    unsafe {
        let _ = proc_reset_footprint_interval(libc::getpid());
    }
}

/// Logs the current physical memory usage at the point of call.
pub fn print_current_memory_usage(label: &str) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        if let Some(usage) = memory_stats() {
            let memory_gib = usage.physical_mem as f64 / BYTES_PER_GIB;
            tracing::debug!(
                label = label,
                usage = %format_memory_size(memory_gib),
                "current memory usage"
            );
        } else {
            tracing::debug!(label = label, "memory stats unavailable");
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn memory_span_start_end_records_delta() {
        start_memory_tracing_span("test_span_lifecycle");
        end_memory_tracing_span("test_span_lifecycle");
        let map = MEMORY_DELTA_MAP.lock().unwrap();
        assert!(map.contains_key("test_span_lifecycle"));
    }

    #[test]
    fn duplicate_span_warns_without_panic() {
        start_memory_tracing_span("test_span_dup");
        start_memory_tracing_span("test_span_dup");
    }

    #[test]
    fn end_without_start_warns_without_panic() {
        end_memory_tracing_span("test_span_nonexistent");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn footprint_interval_peak_keeps_an_unmapped_transient() {
        const BYTES: usize = 256 << 20;
        reset_footprint_interval();
        let before = phys_footprint().unwrap();
        // mmap/munmap rather than Vec: libmalloc keeps freed large blocks
        // resident, so only an unmap guarantees the footprint falls back.
        // SAFETY: an anonymous private mapping of BYTES, written in bounds
        // and unmapped once.
        unsafe {
            let mapping = libc::mmap(
                std::ptr::null_mut(),
                BYTES,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            );
            assert_ne!(mapping, libc::MAP_FAILED);
            std::ptr::write_bytes(mapping.cast::<u8>(), 1, BYTES);
            assert_eq!(libc::munmap(mapping, BYTES), 0);
        }
        let after = phys_footprint().unwrap();
        assert!(after.current_bytes < before.current_bytes + BYTES as u64);
        assert!(after.interval_peak_bytes >= before.current_bytes + BYTES as u64);
        assert!(after.lifetime_peak_bytes >= after.interval_peak_bytes);
        reset_footprint_interval();
        let reset = phys_footprint().unwrap();
        assert!(reset.interval_peak_bytes < before.current_bytes + BYTES as u64);
    }
}
