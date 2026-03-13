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

static MEMORY_USAGE_MAP: LazyLock<Mutex<BTreeMap<&'static str, f64>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
static MEMORY_DELTA_MAP: LazyLock<Mutex<BTreeMap<&'static str, f64>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Records the current physical memory usage at the start of a labeled span.
///
/// # Panics
///
/// Panics if a span with the same label is already open (nested spans need distinct labels).
pub fn start_memory_tracing_span(label: &'static str) {
    let memory_usage = memory_stats().unwrap().physical_mem;
    let mut map = MEMORY_USAGE_MAP.lock().unwrap();
    assert_eq!(
        map.insert(label, memory_usage as f64 / 1_000_000_000.0),
        None,
        "duplicate memory span label: {label}"
    );
}

/// Closes a labeled memory span and records the memory delta (in GB).
///
/// # Panics
///
/// Panics if no span with the given label was previously opened.
pub fn end_memory_tracing_span(label: &'static str) {
    let memory_usage_end = memory_stats().unwrap().physical_mem as f64 / 1_000_000_000.0;
    let mut memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    let memory_usage_start = memory_usage_map
        .remove(label)
        .unwrap_or_else(|| panic!("no open memory span: {label}"));

    let memory_usage_delta = memory_usage_end - memory_usage_start;
    let mut memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
    assert_eq!(memory_delta_map.insert(label, memory_usage_delta), None);
}

/// Logs all collected memory deltas and warns about any unclosed spans.
pub fn report_memory_usage() {
    tracing::info!("================ MEMORY USAGE REPORT ================");

    let memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    for label in memory_usage_map.keys() {
        tracing::warn!("  Unclosed memory tracing span: \"{label}\"");
    }

    let memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
    for (label, delta) in memory_delta_map.iter() {
        if *delta >= 1.0 {
            tracing::info!("  \"{label}\": {delta:.2} GB");
        } else {
            tracing::info!("  \"{}\": {:.2} MB", label, delta * 1000.0);
        }
    }

    tracing::info!("=====================================================");
}

/// Logs the current physical memory usage at the point of call.
pub fn print_current_memory_usage(label: &str) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        if let Some(usage) = memory_stats() {
            let memory_usage_gb = usage.physical_mem as f64 / 1_000_000_000.0;
            if memory_usage_gb >= 1.0 {
                tracing::debug!("\"{label}\" current memory usage: {memory_usage_gb:.2} GB");
            } else {
                tracing::debug!(
                    "\"{}\" current memory usage: {:.2} MB",
                    label,
                    memory_usage_gb * 1000.0
                );
            }
        } else {
            tracing::debug!("Failed to get current memory usage (\"{label}\")");
        }
    }
}
