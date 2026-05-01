//! Profiling and tracing infrastructure for the Jolt proving system.
//!
//! Provides a unified interface for performance analysis across all Jolt crates:
//!
//! - **Tracing subscriber setup** — configures `tracing-chrome` (Perfetto/Chrome JSON)
//!   and `tracing-subscriber` (console output) for the host binary.
//! - **Memory profiling** — tracks memory deltas across proving stages via `memory-stats`.
//! - **System metrics monitoring** (`monitor` feature) — background thread sampling
//!   CPU usage, memory, active cores, and thread count. Outputs structured counter events
//!   compatible with the Perfetto postprocessing script.
//! - **CPU profiling** (`pprof` feature) — scoped `pprof` guards that write `.pb`
//!   flamegraph files on drop.
//! - **Heap flamegraphs** (`allocative` feature) — generates SVG flamegraphs from
//!   `allocative`-instrumented data structures.
//!
//! # Usage
//!
//! Individual crates add `tracing` as a dependency and instrument their functions with
//! `#[tracing::instrument]`. The host binary (e.g. `jolt-zkvm` CLI) depends on
//! `jolt-profiling` to configure the subscriber that captures those spans.
//!
//! ```no_run
//! use jolt_profiling::{setup_tracing, TracingFormat};
//!
//! let _guards = setup_tracing(
//!     &[TracingFormat::Chrome],
//!     "my_benchmark_20260306",
//! );
//! // All tracing spans from any Jolt crate now flow to Perfetto JSON output.
//! ```
//!
//! # Feature Flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `monitor` | Background system metrics sampling (CPU, memory, cores) |
//! | `pprof` | Scoped CPU profiling via `pprof` with `.pb` output |
//! | `allocative` | Heap flamegraph generation from `allocative`-instrumented types |
//!
//! # Dependency Position
//!
//! This is a leaf crate — imported by host binaries and benchmarks.
//! Library crates depend only on `tracing` for instrumentation.

pub mod setup;

#[cfg(not(target_arch = "wasm32"))]
pub mod memory;

#[cfg(all(not(target_arch = "wasm32"), feature = "monitor"))]
pub mod monitor;

mod pprof_guard;

#[cfg(feature = "allocative")]
pub mod flamegraph;
#[cfg(feature = "allocative")]
pub use flamegraph::{print_data_structure_heap_usage, write_flamegraph_svg};

mod units;

pub use setup::{setup_tracing, TracingFormat, TracingGuards};
pub use units::{format_memory_size, BYTES_PER_GIB, BYTES_PER_MIB};

#[cfg(not(target_arch = "wasm32"))]
pub use memory::{
    end_memory_tracing_span, print_current_memory_usage, report_memory_usage,
    start_memory_tracing_span,
};

#[cfg(target_arch = "wasm32")]
pub fn start_memory_tracing_span(_label: &'static str) {}

#[cfg(target_arch = "wasm32")]
pub fn end_memory_tracing_span(_label: &'static str) {}

#[cfg(target_arch = "wasm32")]
pub fn report_memory_usage() {}

#[cfg(target_arch = "wasm32")]
pub fn print_current_memory_usage(_label: &str) {}

#[cfg(all(not(target_arch = "wasm32"), feature = "monitor"))]
pub use monitor::MetricsMonitor;

#[cfg(all(target_arch = "wasm32", feature = "monitor"))]
#[must_use = "monitor stops when dropped"]
pub struct MetricsMonitor;

#[cfg(all(target_arch = "wasm32", feature = "monitor"))]
impl MetricsMonitor {
    pub fn start(_interval_secs: f64) -> Self {
        Self
    }
}

pub use pprof_guard::PprofGuard;
