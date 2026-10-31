//! Span-boundary sampling of process RSS and instructions retired.
//!
//! [`SpanBoundaryLayer`] samples at every span enter and exit, the instance
//! boundaries the chrome trace records:
//!
//! - **Instructions** for every span entered on the thread that opened the
//!   process counter (the private `instructions` module), however the span
//!   was declared. Each enter/exit pair adds its process-wide delta to its
//!   label's [`SpanInstructions`]; spans on other threads are skipped,
//!   because concurrent siblings would share their deltas.
//! - **RSS** for the prover-stage spans (`prove_stage0`..`prove_stage8` plus
//!   the `jolt_prover::prove` root) on any thread, recorded with the same
//!   instance's instruction count as a [`StageBoundaryRow`] and emitted as a
//!   `stage_boundary` tracing event, so a Chrome/Perfetto trace carries the
//!   samples as an instant event next to the stage's slice.
//!
//! Boundary sampling attributes *retained* growth per stage; short
//! intra-stage spikes are invisible to it. The headline number for a run is
//! [`peak_rss_bytes`](crate::memory::peak_rss_bytes), the kernel-maintained
//! high-water mark. Instruction counts are process-wide: a span's count
//! covers every counted thread during the span's interval, which is the
//! instruction analogue of its wallclock. A span entered again before it
//! exits keeps only its innermost sample, so its label records fewer
//! instances than the trace and the summary reports no total for it.

use std::collections::BTreeMap;
use std::sync::Mutex;

use memory_stats::memory_stats;
use tracing::span::Id;
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::instructions::{read_process_counter, InstructionReading};
use crate::taxonomy::ROOT_SPAN;
use crate::units::{format_memory_size, BYTES_PER_GIB};

#[derive(Clone, Copy)]
struct EnterSample {
    /// Stage and root spans only.
    rss_bytes: Option<u64>,
    instructions: Option<InstructionReading>,
}

/// One stage or root span instance's boundary samples.
#[derive(Clone, Copy, Debug)]
pub struct StageBoundaryRow {
    pub stage: &'static str,
    pub rss_open_bytes: u64,
    pub rss_close_bytes: u64,
    /// User-space instructions the process retired between enter and exit;
    /// `None` off the counter's opening thread, without a hardware counter,
    /// or when the kernel multiplexed it.
    pub instructions: Option<u64>,
}

/// One span label's instruction totals over its instances entered on the
/// counter's opening thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpanInstructions {
    /// Sampled enter/exit pairs.
    pub instances: u64,
    /// Summed process-wide deltas; `None` once any instance's count was
    /// unavailable.
    pub instructions: Option<u64>,
}

impl SpanInstructions {
    fn add(&mut self, instructions: Option<u64>) {
        self.instructions = if self.instances == 0 {
            instructions
        } else {
            self.instructions.zip(instructions).map(|(sum, n)| sum + n)
        };
        self.instances += 1;
    }
}

/// Everything the layer recorded since the last drain.
#[derive(Clone, Debug, Default)]
pub struct BoundarySamples {
    /// Stage and root span instances, in exit order.
    pub stage_rows: Vec<StageBoundaryRow>,
    pub span_instructions: BTreeMap<&'static str, SpanInstructions>,
}

/// Cap on retained stage rows: a prove records ~11 rows, so this covers
/// ~90 undrained proves while bounding the global log in a long-lived
/// process that installs the layer but never calls
/// [`take_boundary_samples`]. Label totals are bounded by the set of span
/// names.
const MAX_STAGE_BOUNDARY_ROWS: usize = 1024;

/// The global sample log plus a saturation marker, so overflow warns once
/// per drain instead of per dropped row.
struct BoundaryLog {
    samples: BoundarySamples,
    warned_full: bool,
}

static BOUNDARY_LOG: Mutex<BoundaryLog> = Mutex::new(BoundaryLog {
    samples: BoundarySamples {
        stage_rows: Vec::new(),
        span_instructions: BTreeMap::new(),
    },
    warned_full: false,
});

fn is_stage(name: &str) -> bool {
    name.starts_with("prove_stage") || name == ROOT_SPAN
}

/// A `tracing_subscriber` layer sampling instructions retired at the
/// boundaries of every span on the counter's opening thread, and process
/// RSS at stage-span boundaries. Installed by
/// [`setup_tracing`](crate::setup_tracing).
pub struct SpanBoundaryLayer;

impl<S> Layer<S> for SpanBoundaryLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let rss_bytes = if is_stage(span.name()) {
            memory_stats().map(|stats| stats.physical_mem as u64)
        } else {
            None
        };
        // Read last on enter and first on exit, so the span's count
        // excludes this layer's own bookkeeping.
        let instructions = read_process_counter();
        if rss_bytes.is_none() && instructions.is_none() {
            return;
        }
        let _ = span.extensions_mut().replace(EnterSample {
            rss_bytes,
            instructions,
        });
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let instructions_at_exit = read_process_counter();
        let Some(span) = ctx.span(id) else { return };
        if instructions_at_exit.is_none() && !is_stage(span.name()) {
            return;
        }
        let Some(entered) = span.extensions_mut().remove::<EnterSample>() else {
            return;
        };
        let instructions = instructions_at_exit
            .zip(entered.instructions)
            .and_then(|(exit, enter)| exit.instructions_since(enter));
        let row = entered.rss_bytes.and_then(|rss_open_bytes| {
            let stats = memory_stats()?;
            Some(StageBoundaryRow {
                stage: span.name(),
                rss_open_bytes,
                rss_close_bytes: stats.physical_mem as u64,
                instructions,
            })
        });
        if let Some(row) = row {
            tracing::info!(
                stage = row.stage,
                rss_open_gib = row.rss_open_bytes as f64 / BYTES_PER_GIB,
                rss_close_gib = row.rss_close_bytes as f64 / BYTES_PER_GIB,
                instructions = row.instructions,
                "stage_boundary"
            );
        }

        let mut log = BOUNDARY_LOG.lock().unwrap_or_else(|e| e.into_inner());
        if entered.instructions.is_some() {
            log.samples
                .span_instructions
                .entry(span.name())
                .or_default()
                .add(instructions);
        }
        let Some(row) = row else { return };
        if log.samples.stage_rows.len() >= MAX_STAGE_BOUNDARY_ROWS {
            if !log.warned_full {
                log.warned_full = true;
                tracing::warn!(
                    cap = MAX_STAGE_BOUNDARY_ROWS,
                    "stage-boundary row log is full; dropping rows until it is drained"
                );
            }
            return;
        }
        log.samples.stage_rows.push(row);
    }
}

/// Drain and return the samples recorded so far.
pub fn take_boundary_samples() -> BoundarySamples {
    let mut log = BOUNDARY_LOG.lock().unwrap_or_else(|e| e.into_inner());
    log.warned_full = false;
    std::mem::take(&mut log.samples)
}

/// Print the recorded per-stage RSS and instruction table to stdout and
/// clear the log. Call once at the end of a benchmark run.
#[expect(
    clippy::print_stdout,
    reason = "benchmark-harness reporting; stdout is the deliverable"
)]
pub fn report_stage_boundaries() {
    let rows = take_boundary_samples().stage_rows;
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
