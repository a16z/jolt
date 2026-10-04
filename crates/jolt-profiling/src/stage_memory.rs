//! Per-stage RSS tracking driven by span lifecycle.
//!
//! [`StageMemoryLayer`] watches the prover-stage spans
//! (`prove_stage0`..`prove_stage8` plus the `jolt_prover::prove` root), samples
//! the process RSS
//! when each span opens and closes, and records the rows for
//! [`report_stage_memory`]. It also emits a
//! `stage_rss` tracing event at every close, so a Chrome/Perfetto trace
//! carries the boundary RSS as an instant event next to the stage's slice.
//!
//! Boundary sampling attributes *retained* growth per stage; short
//! intra-stage spikes are invisible to it. Where the kernel exposes it
//! (macOS), each row also carries the physical footprint at both boundaries
//! and its kernel-maintained high-water mark inside the span, which no
//! transient can escape, plus the Metal device's allocated bytes at both
//! boundaries when a probe is registered ([`set_device_memory_probe`]).

use std::sync::{Mutex, OnceLock};

use memory_stats::memory_stats;
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::memory::{phys_footprint, reset_footprint_interval, FootprintSample};
use crate::taxonomy::ROOT_SPAN;
use crate::units::{format_memory_size, BYTES_PER_GIB};

/// One tracked span's samples at open, parked in the span's extensions.
#[derive(Clone, Copy)]
struct RssAtOpen {
    rss: u64,
    footprint: Option<FootprintAtOpen>,
}

#[derive(Clone, Copy)]
struct FootprintAtOpen {
    open: u64,
    gap_peak: u64,
    device_open: Option<u64>,
}

/// One closed stage span's boundary RSS samples, in close order.
#[derive(Clone, Copy, Debug)]
pub struct StageMemoryRow {
    pub stage: &'static str,
    pub rss_open_bytes: u64,
    pub rss_close_bytes: u64,
    pub footprint: Option<StageFootprint>,
}

/// Physical footprint of one tracked span, in bytes.
#[derive(Clone, Copy, Debug)]
pub struct StageFootprint {
    pub open: u64,
    pub close: u64,
    /// High-water mark inside the span; the root span reports the process
    /// lifetime peak.
    pub peak: u64,
    /// High-water mark between the previous tracked boundary and this open.
    pub gap_peak: u64,
    pub device_open: Option<u64>,
    pub device_close: Option<u64>,
}

static DEVICE_MEMORY_PROBE: OnceLock<fn() -> u64> = OnceLock::new();

/// Registers the accelerator's allocated-bytes counter sampled at every
/// tracked boundary. The first registration wins.
pub fn set_device_memory_probe(probe: fn() -> u64) {
    let _ = DEVICE_MEMORY_PROBE.set(probe);
}

fn device_bytes() -> Option<u64> {
    DEVICE_MEMORY_PROBE.get().map(|probe| probe())
}

/// Reads the interval high-water mark, then restarts it: tracked spans
/// partition the timeline into consecutive intervals.
fn take_footprint() -> Option<FootprintSample> {
    let sample = phys_footprint()?;
    reset_footprint_interval();
    Some(sample)
}

/// Cap on retained rows: a prove records ~11 rows, so this covers ~90
/// undrained proves while bounding the global log in a long-lived process
/// that installs the layer but never calls [`take_stage_memory_rows`].
const MAX_STAGE_MEMORY_ROWS: usize = 1024;

/// The global row log plus a saturation marker, so overflow warns once per
/// drain instead of per dropped row.
struct RowLog {
    rows: Vec<StageMemoryRow>,
    warned_full: bool,
}

static STAGE_MEMORY_ROWS: Mutex<RowLog> = Mutex::new(RowLog {
    rows: Vec::new(),
    warned_full: false,
});

/// The stage spans worth boundary-sampling.
fn tracked(name: &str) -> bool {
    name.starts_with("prove_stage") || name == ROOT_SPAN
}

/// A `tracing_subscriber` layer sampling process RSS at stage-span
/// boundaries. Installed by [`setup_tracing`](crate::setup_tracing); inert
/// (two string comparisons per span) for every other span.
pub struct StageMemoryLayer;

impl<S> Layer<S> for StageMemoryLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if !tracked(span.name()) {
            return;
        }
        let Some(stats) = memory_stats() else { return };
        let footprint = take_footprint().map(|sample| FootprintAtOpen {
            open: sample.current_bytes,
            gap_peak: sample.interval_peak_bytes,
            device_open: device_bytes(),
        });
        span.extensions_mut().insert(RssAtOpen {
            rss: stats.physical_mem as u64,
            footprint,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        if !tracked(span.name()) {
            return;
        }
        let opened = span.extensions().get::<RssAtOpen>().copied();
        let Some(opened) = opened else {
            return;
        };
        let Some(stats) = memory_stats() else { return };
        let footprint =
            opened
                .footprint
                .zip(take_footprint())
                .map(|(open, close)| StageFootprint {
                    open: open.open,
                    close: close.current_bytes,
                    peak: if span.name() == ROOT_SPAN {
                        close.lifetime_peak_bytes
                    } else {
                        close.interval_peak_bytes
                    },
                    gap_peak: open.gap_peak,
                    device_open: open.device_open,
                    device_close: device_bytes(),
                });
        let row = StageMemoryRow {
            stage: span.name(),
            rss_open_bytes: opened.rss,
            rss_close_bytes: stats.physical_mem as u64,
            footprint,
        };
        // An instant event for the Chrome/Perfetto trace, anchoring the
        // boundary samples next to the stage's slice.
        tracing::info!(
            stage = row.stage,
            rss_open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB,
            rss_close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB,
            footprint_open_bytes = footprint.map(|f| f.open),
            footprint_close_bytes = footprint.map(|f| f.close),
            footprint_peak_bytes = footprint.map(|f| f.peak),
            footprint_gap_peak_bytes = footprint.map(|f| f.gap_peak),
            device_open_bytes = footprint.and_then(|f| f.device_open),
            device_close_bytes = footprint.and_then(|f| f.device_close),
            "stage_rss"
        );
        let mut log = STAGE_MEMORY_ROWS.lock().unwrap_or_else(|e| e.into_inner());
        if log.rows.len() >= MAX_STAGE_MEMORY_ROWS {
            if !log.warned_full {
                log.warned_full = true;
                tracing::warn!(
                    cap = MAX_STAGE_MEMORY_ROWS,
                    "stage-memory row log is full; dropping rows until it is drained"
                );
            }
            return;
        }
        log.rows.push(row);
    }
}

/// Drain and return the rows recorded so far, in span-close order.
pub fn take_stage_memory_rows() -> Vec<StageMemoryRow> {
    let mut log = STAGE_MEMORY_ROWS.lock().unwrap_or_else(|e| e.into_inner());
    log.warned_full = false;
    std::mem::take(&mut log.rows)
}

/// Print the recorded per-stage RSS table to stdout and clear the log.
/// Call once at the end of a benchmark run.
#[expect(
    clippy::print_stdout,
    reason = "benchmark-harness reporting; stdout is the deliverable"
)]
pub fn report_stage_memory() {
    let rows = take_stage_memory_rows();
    if rows.is_empty() {
        return;
    }
    println!("Per-stage RSS at span boundaries (start → end, Δ retained):");
    for row in rows {
        let open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB;
        let close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB;
        println!(
            "  {:<14} {:>10} → {:>10}  (Δ {:>10})",
            row.stage,
            format_memory_size(open_gib),
            format_memory_size(close_gib),
            format_memory_size(close_gib - open_gib),
        );
    }
}

/// Print the per-stage physical-footprint table (exact bytes) to stdout
/// without draining the row log. No output where footprints are unsupported.
#[expect(
    clippy::print_stdout,
    reason = "benchmark-harness reporting; stdout is the deliverable"
)]
pub fn report_stage_footprint() {
    let rows = STAGE_MEMORY_ROWS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .rows
        .clone();
    if rows.iter().all(|row| row.footprint.is_none()) {
        return;
    }
    println!(
        "Per-stage physical footprint, bytes (peak: kernel high-water mark inside the stage, \
         lifetime for the root; gap: since the previous boundary; device: Metal allocated):"
    );
    println!(
        "  {:<24} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}",
        "stage", "open", "close", "peak", "gap_peak", "device_open", "device_close"
    );
    let bytes = |value: Option<u64>| value.map_or_else(|| "-".to_owned(), |v| v.to_string());
    for row in rows {
        let Some(footprint) = row.footprint else {
            continue;
        };
        println!(
            "  {:<24} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}",
            row.stage,
            footprint.open,
            footprint.close,
            footprint.peak,
            footprint.gap_peak,
            bytes(footprint.device_open),
            bytes(footprint.device_close),
        );
    }
}
