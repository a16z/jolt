//! Per-stage RSS and instruction counts driven by span lifecycle.
//!
//! [`StageBoundaryLayer`] watches the prover-stage spans
//! (`prove_stage0`..`prove_stage8` plus the `jolt_prover::prove` root),
//! samples the process RSS and the process instruction counter (the
//! private `instructions` module) when each span opens and closes,
//! and records the rows for [`report_stage_boundaries`]. It also emits a
//! `stage_boundary` tracing event at every close, so a Chrome/Perfetto trace
//! carries the boundary samples as an instant event next to the stage's
//! slice.
//!
//! Boundary sampling attributes *retained* growth per stage; short
//! intra-stage spikes are invisible to it. The headline number for a run is
//! [`peak_rss_bytes`](crate::memory::peak_rss_bytes), the kernel-maintained
//! high-water mark. Instruction counts are process-wide: a stage's count
//! covers every counted thread during the stage's interval, which is the
//! instruction analogue of its wallclock.

use std::sync::Mutex;

use memory_stats::memory_stats;
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::instructions::{read_process_counter, InstructionReading};
use crate::taxonomy::ROOT_SPAN;
use crate::units::{format_memory_size, BYTES_PER_GIB};

#[derive(Clone, Copy)]
struct OpenSample {
    rss_bytes: u64,
    instructions: Option<InstructionReading>,
}

/// One closed stage span's boundary samples, in close order.
#[derive(Clone, Copy, Debug)]
pub struct StageBoundaryRow {
    pub stage: &'static str,
    pub rss_open_bytes: u64,
    pub rss_close_bytes: u64,
    /// User-space instructions the process retired while the span was open;
    /// `None` without a hardware counter or when the kernel multiplexed it.
    pub instructions: Option<u64>,
}

/// Cap on retained rows: a prove records ~11 rows, so this covers ~90
/// undrained proves while bounding the global log in a long-lived process
/// that installs the layer but never calls [`take_stage_boundary_rows`].
const MAX_STAGE_BOUNDARY_ROWS: usize = 1024;

/// The global row log plus a saturation marker, so overflow warns once per
/// drain instead of per dropped row.
struct RowLog {
    rows: Vec<StageBoundaryRow>,
    warned_full: bool,
}

static STAGE_BOUNDARY_ROWS: Mutex<RowLog> = Mutex::new(RowLog {
    rows: Vec::new(),
    warned_full: false,
});

fn tracked(name: &str) -> bool {
    name.starts_with("prove_stage") || name == ROOT_SPAN
}

/// A `tracing_subscriber` layer sampling process RSS and instructions
/// retired at stage-span boundaries. Installed by
/// [`setup_tracing`](crate::setup_tracing); inert (two string comparisons
/// per span) for every other span.
pub struct StageBoundaryLayer;

impl<S> Layer<S> for StageBoundaryLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if !tracked(span.name()) {
            return;
        }
        let Some(stats) = memory_stats() else { return };
        // Read last on open and first on close, so the stage's count
        // excludes this layer's own bookkeeping.
        let instructions = read_process_counter();
        span.extensions_mut().insert(OpenSample {
            rss_bytes: stats.physical_mem as u64,
            instructions,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        if !tracked(span.name()) {
            return;
        }
        let instructions_at_close = read_process_counter();
        let opened = span.extensions().get::<OpenSample>().copied();
        let Some(opened) = opened else {
            return;
        };
        let Some(stats) = memory_stats() else { return };
        let row = StageBoundaryRow {
            stage: span.name(),
            rss_open_bytes: opened.rss_bytes,
            rss_close_bytes: stats.physical_mem as u64,
            instructions: instructions_at_close
                .zip(opened.instructions)
                .and_then(|(close, open)| close.instructions_since(open)),
        };
        tracing::info!(
            stage = row.stage,
            rss_open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB,
            rss_close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB,
            instructions = row.instructions,
            "stage_boundary"
        );
        let mut log = STAGE_BOUNDARY_ROWS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if log.rows.len() >= MAX_STAGE_BOUNDARY_ROWS {
            if !log.warned_full {
                log.warned_full = true;
                tracing::warn!(
                    cap = MAX_STAGE_BOUNDARY_ROWS,
                    "stage-boundary row log is full; dropping rows until it is drained"
                );
            }
            return;
        }
        log.rows.push(row);
    }
}

/// Drain and return the rows recorded so far, in span-close order.
pub fn take_stage_boundary_rows() -> Vec<StageBoundaryRow> {
    let mut log = STAGE_BOUNDARY_ROWS
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    log.warned_full = false;
    std::mem::take(&mut log.rows)
}

/// Print the recorded per-stage RSS and instruction table to stdout and
/// clear the log. Call once at the end of a benchmark run.
#[expect(
    clippy::print_stdout,
    reason = "benchmark-harness reporting; stdout is the deliverable"
)]
pub fn report_stage_boundaries() {
    let rows = take_stage_boundary_rows();
    if rows.is_empty() {
        return;
    }
    println!(
        "Per-stage boundaries (RSS start → end, Δ retained; user-space instructions retired):"
    );
    for row in rows {
        let open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB;
        let close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB;
        let instructions = row.instructions.map_or_else(
            || "n/a".to_string(),
            |count| format!("{:.3}G", count as f64 / 1e9),
        );
        println!(
            "  {:<14} {:>10} → {:>10}  (Δ {:>10})  {:>10}",
            row.stage,
            format_memory_size(open_gib),
            format_memory_size(close_gib),
            format_memory_size(close_gib - open_gib),
            instructions,
        );
    }
}
