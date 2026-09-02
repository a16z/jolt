//! Per-stage RSS tracking driven by span lifecycle.
//!
//! [`StageMemoryLayer`] watches the prover-stage spans both provers emit
//! (`prove_stage0`..`prove_stage8`, plus the whole-run roots
//! `jolt_prover::prove` / `prove_parts` / `E2E`), samples the process RSS
//! when each span opens and closes, and records the rows for
//! [`report_stage_memory`]. It also emits a
//! `stage_rss` tracing event at every close, so a Chrome/Perfetto trace
//! carries the boundary RSS as an instant event next to the stage's slice.
//!
//! Boundary sampling attributes *retained* growth per stage; short
//! intra-stage spikes are invisible to it. The headline number for a run is
//! [`peak_rss_bytes`](crate::memory::peak_rss_bytes), the kernel-maintained
//! high-water mark.

use std::sync::Mutex;

use memory_stats::memory_stats;
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::memory::phys_footprint;
use crate::units::{format_memory_size, BYTES_PER_GIB};

/// One tracked span's RSS (and footprint, where the OS ledger exists) at
/// open, parked in the span's extensions.
#[derive(Clone, Copy)]
struct RssAtOpen {
    rss: u64,
    footprint: Option<u64>,
}

/// One closed stage span's boundary memory samples, in close order.
/// `footprint_*` is macOS `phys_footprint` (see
/// [`crate::memory::PhysFootprint`]); `None` off macOS.
#[derive(Clone, Copy, Debug)]
pub struct StageMemoryRow {
    pub stage: &'static str,
    pub rss_open_bytes: u64,
    pub rss_close_bytes: u64,
    pub footprint_open_bytes: Option<u64>,
    pub footprint_close_bytes: Option<u64>,
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

/// The stage spans worth boundary-sampling: the per-stage prover recipes
/// (modular `prove_stage0`..`prove_stage8`, legacy `prove_stage1`..) and the
/// whole-run roots (modular `jolt_prover::prove`, legacy `prove_parts`, both
/// harnesses' `E2E`).
fn tracked(name: &str) -> bool {
    name.starts_with("prove_stage")
        || name == crate::taxonomy::ROOT_SPAN
        || name == "prove_parts"
        || name == "E2E"
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
        span.extensions_mut().insert(RssAtOpen {
            rss: stats.physical_mem as u64,
            footprint: phys_footprint().map(|fp| fp.current_bytes),
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        if !tracked(span.name()) {
            return;
        }
        let opened = span.extensions().get::<RssAtOpen>().copied();
        let Some(open) = opened else {
            return;
        };
        let Some(stats) = memory_stats() else { return };
        let row = StageMemoryRow {
            stage: span.name(),
            rss_open_bytes: open.rss,
            rss_close_bytes: stats.physical_mem as u64,
            footprint_open_bytes: open.footprint,
            footprint_close_bytes: phys_footprint().map(|fp| fp.current_bytes),
        };
        // An instant event for the Chrome/Perfetto trace, anchoring the
        // boundary RSS next to the stage's slice.
        tracing::info!(
            stage = row.stage,
            rss_open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB,
            rss_close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB,
            footprint_open_gib = row.footprint_open_bytes.unwrap_or(0) as f64 / BYTES_PER_GIB,
            footprint_close_gib = row.footprint_close_bytes.unwrap_or(0) as f64 / BYTES_PER_GIB,
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
    println!("Per-stage RSS [footprint] at span boundaries (start → end, Δ retained):");
    for row in rows {
        let open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB;
        let close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB;
        let footprint = match (row.footprint_open_bytes, row.footprint_close_bytes) {
            (Some(open), Some(close)) => format!(
                "  [{} → {}]",
                format_memory_size(open as f64 / BYTES_PER_GIB),
                format_memory_size(close as f64 / BYTES_PER_GIB),
            ),
            _ => String::new(),
        };
        println!(
            "  {:<14} {:>10} → {:>10}  (Δ {:>10}){footprint}",
            row.stage,
            format_memory_size(open_gib),
            format_memory_size(close_gib),
            format_memory_size(close_gib - open_gib),
        );
    }
}
