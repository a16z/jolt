//! Memory profiling utilities.
//!
//! Tracks physical memory deltas across labeled spans. Call
//! [`start_memory_tracing_span`] before the section and
//! [`end_memory_tracing_span`] after, then [`report_memory_usage`] to
//! log all collected deltas.

#[cfg(target_os = "macos")]
use libc::{rusage_info_v4, RUSAGE_INFO_V4};
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

/// The kernel's `rusage_info_v4` snapshot of this process.
#[cfg(target_os = "macos")]
fn rusage_info() -> Option<rusage_info_v4> {
    // SAFETY: on success (return value 0) proc_pid_rusage writes a complete
    // `rusage_info_v4` for the requested flavor into the provided storage.
    unsafe {
        let mut info: rusage_info_v4 = std::mem::zeroed();
        let status = libc::proc_pid_rusage(libc::getpid(), RUSAGE_INFO_V4, (&raw mut info).cast());
        (status == 0).then_some(info)
    }
}

/// Current physical footprint in bytes, sampled from the kernel's
/// `ri_phys_footprint` counter. Unlike RSS it includes pages macOS has
/// compressed under memory pressure. `None` off macOS; no separate footprint counter is provided.
pub fn current_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        rusage_info().map(|info| info.ri_phys_footprint)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Process-lifetime peak physical footprint in bytes.
///
/// On macOS, memory pressure makes the kernel compress cold pages, which
/// drop out of RSS: [`peak_rss_bytes`] can under-report a compressed run by
/// half. `ri_lifetime_max_phys_footprint` counts them, and is the figure
/// `/usr/bin/time -l` prints as "peak memory footprint". Elsewhere this falls back to [`peak_rss_bytes`]; it does not account
/// for swapped or compressed pages.
pub fn peak_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        rusage_info().map(|info| info.ri_lifetime_max_phys_footprint)
    }
    #[cfg(not(target_os = "macos"))]
    {
        peak_rss_bytes()
    }
}

/// The process-lifetime memory high-water marks a profiled run reports.
/// Sample once, right after the workload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeakMemory {
    /// [`peak_rss_bytes`].
    pub rss_bytes: Option<u64>,
    /// [`peak_footprint_bytes`].
    pub footprint_bytes: Option<u64>,
}

impl PeakMemory {
    pub fn sample() -> Self {
        Self {
            rss_bytes: peak_rss_bytes(),
            footprint_bytes: peak_footprint_bytes(),
        }
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

    #[cfg(unix)]
    #[test]
    fn footprint_covers_a_live_allocation() {
        const BYTES: usize = 64 << 20;
        let block = std::hint::black_box(vec![1u8; BYTES]);
        assert!(peak_footprint_bytes().unwrap() >= BYTES as u64);
        #[cfg(target_os = "macos")]
        assert!(current_footprint_bytes().unwrap() >= BYTES as u64);
        drop(block);
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
