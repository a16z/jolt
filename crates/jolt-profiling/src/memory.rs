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
}
