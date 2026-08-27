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

/// The kernel's memory-ledger view of the process, from
/// `proc_pid_rusage(RUSAGE_INFO_V4)`.
///
/// `phys_footprint` is what macOS actually bills the process (anonymous
/// dirty + compressed + IOKit/Metal wired mappings) and what the jetsam
/// limit acts on; RSS over-counts resident-but-clean file pages and
/// under-counts compressed ones, so the two diverge exactly in the
/// GPU-heavy runs where memory matters. Both fields are kernel-maintained:
/// `lifetime_peak` cannot miss short spikes.
#[derive(Clone, Copy, Debug)]
pub struct PhysFootprint {
    pub current_bytes: u64,
    pub lifetime_peak_bytes: u64,
}

/// Sample the process's physical footprint (current + lifetime peak).
/// Returns `None` off macOS or if the syscall fails.
#[cfg(target_os = "macos")]
pub fn phys_footprint() -> Option<PhysFootprint> {
    // `struct rusage_info_v4` from <sys/resource.h>: 16 UUID bytes then 35
    // u64 ledger fields. Declared locally (libSystem exports the symbol;
    // the layout is ABI-stable — flavors are versioned by definition).
    #[repr(C)]
    struct RusageInfoV4 {
        ri_uuid: [u8; 16],
        ri_fields: [u64; 35],
    }
    /// `ri_phys_footprint` / `ri_lifetime_max_phys_footprint` zero-based
    /// positions in the post-UUID u64 ledger, pinned by the
    /// `footprint_tracks_dirty_memory` test against a known allocation.
    const PHYS_FOOTPRINT: usize = 7;
    const LIFETIME_MAX_PHYS_FOOTPRINT: usize = 28;
    const RUSAGE_INFO_V4: libc::c_int = 4;
    extern "C" {
        fn proc_pid_rusage(
            pid: libc::c_int,
            flavor: libc::c_int,
            buffer: *mut libc::c_void,
        ) -> libc::c_int;
    }
    let mut info = std::mem::MaybeUninit::<RusageInfoV4>::zeroed();
    // SAFETY: RUSAGE_INFO_V4 fills exactly a `rusage_info_v4`, and the
    // buffer is sized as one; the call writes nothing on failure.
    let ok = unsafe {
        proc_pid_rusage(
            std::process::id() as libc::c_int,
            RUSAGE_INFO_V4,
            info.as_mut_ptr().cast(),
        )
    } == 0;
    if !ok {
        return None;
    }
    // SAFETY: zeroed + fully written by the successful call above.
    let info = unsafe { info.assume_init() };
    Some(PhysFootprint {
        current_bytes: info.ri_fields[PHYS_FOOTPRINT],
        lifetime_peak_bytes: info.ri_fields[LIFETIME_MAX_PHYS_FOOTPRINT],
    })
}

/// Physical footprint is a macOS ledger; `None` elsewhere.
#[cfg(not(target_os = "macos"))]
pub fn phys_footprint() -> Option<PhysFootprint> {
    None
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

    /// Pins the `rusage_info_v4` ledger offsets: dirtying 256 MiB must move
    /// `phys_footprint` by roughly that much (and never move it by garbage),
    /// and the lifetime peak must dominate the current value.
    #[test]
    #[cfg(target_os = "macos")]
    fn footprint_tracks_dirty_memory() {
        const DIRTY: usize = 256 << 20;
        let before = phys_footprint().unwrap();
        let block = vec![1u8; DIRTY];
        let after = phys_footprint().unwrap();
        assert!(after.lifetime_peak_bytes >= after.current_bytes);
        let grown = after.current_bytes.saturating_sub(before.current_bytes);
        assert!(
            grown > (DIRTY / 2) as u64 && grown < (4 * DIRTY) as u64,
            "footprint delta {grown} not in range for a {DIRTY}-byte dirty block \
             (ledger offsets wrong?)"
        );
        drop(block);
    }

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
}
