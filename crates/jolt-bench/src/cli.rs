use std::path::PathBuf;

use clap::Parser;

use crate::programs::Program;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum StackSelection {
    Core,
    Modular,
    #[default]
    Both,
}

#[derive(Debug, Parser)]
#[command(name = "jolt-bench", about = "Benchmark jolt-core vs modular stack")]
pub struct Cli {
    /// Canonical program to benchmark.
    #[arg(long)]
    pub program: Program,

    /// Which stack(s) to measure.
    #[arg(long, value_enum, default_value_t = StackSelection::Both)]
    pub stack: StackSelection,

    /// Number of measured iterations (medianed in the output).
    #[arg(long, default_value_t = 3)]
    pub iters: usize,

    /// Number of warmup iterations run before measurement (discarded).
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// Write the JSON report to this path. Prints to stdout if absent.
    #[arg(long)]
    pub json: Option<PathBuf>,

    /// Baseline JSON report to compare against. If set, the run exits
    /// non-zero when any modular metric exceeds its baseline `core` value
    /// by more than `--threshold`.
    #[arg(long)]
    pub baseline: Option<PathBuf>,

    /// Regression threshold (multiplier). `1.05` = allow 5% regression.
    #[arg(long, default_value_t = 1.05, requires = "baseline")]
    pub threshold: f64,

    /// Override `max_trace_length` to `1 << log_t`. When unset, the bench
    /// uses its built-in default (`1 << 16`). Actual prover work is
    /// determined by the guest program's execution length (padded to a
    /// power of two ≤ `max_trace_length`), so shrinking this only lowers
    /// the preprocessing cap — it does not force larger traces.
    #[arg(long)]
    pub log_t: Option<usize>,

    /// Emit a Chrome/Perfetto trace of the run to
    /// `benchmark-runs/perfetto_traces/<name>.json`. View at
    /// <https://ui.perfetto.dev/>. The modular stack's prover is
    /// instrumented per-op; core prove is a single span. When set, the
    /// bench forces `--iters 1 --warmup 0` to keep the trace readable.
    #[arg(long, value_name = "NAME")]
    pub trace_chrome: Option<String>,
}
