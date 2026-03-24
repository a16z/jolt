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
/// Logs a warning and returns without recording if memory stats are unavailable.
///
/// # Panics
///
/// Panics if a span with the same label is already open (nested spans need distinct labels).
pub fn start_memory_tracing_span(label: &'static str) {
    let Some(stats) = memory_stats() else {
        tracing::warn!(
            span = label,
            "memory stats unavailable, skipping span start"
        );
        return;
    };
    let memory_gib = stats.physical_mem as f64 / BYTES_PER_GIB;
    let mut map = MEMORY_USAGE_MAP.lock().unwrap();
    assert_eq!(
        map.insert(label, memory_gib),
        None,
        "duplicate memory span label: {label}"
    );
}

/// Closes a labeled memory span and records the memory delta (in GiB).
///
/// Logs a warning and returns without recording if memory stats are unavailable.
///
/// # Panics
///
/// Panics if no span with the given label was previously opened.
pub fn end_memory_tracing_span(label: &'static str) {
    let Some(stats) = memory_stats() else {
        tracing::warn!(span = label, "memory stats unavailable, skipping span end");
        return;
    };
    let memory_gib_end = stats.physical_mem as f64 / BYTES_PER_GIB;
    let mut memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    let memory_gib_start = memory_usage_map
        .remove(label)
        .unwrap_or_else(|| panic!("no open memory span: {label}"));

    let delta = memory_gib_end - memory_gib_start;
    let mut memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
    assert_eq!(memory_delta_map.insert(label, delta), None);
}

/// Logs all collected memory deltas and warns about any unclosed spans.
pub fn report_memory_usage() {
    let memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    for label in memory_usage_map.keys() {
        tracing::warn!(span = label, "unclosed memory tracing span");
    }

    let memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
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
    #[should_panic(expected = "duplicate memory span label")]
    fn duplicate_span_label_panics() {
        start_memory_tracing_span("test_span_dup");
        start_memory_tracing_span("test_span_dup");
    }

    #[test]
    #[should_panic(expected = "no open memory span")]
    fn end_without_start_panics() {
        end_memory_tracing_span("test_span_nonexistent");
    }
}
