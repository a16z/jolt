//! Heap snapshot capture from `allocative`-instrumented data structures.
//!
//! Snapshots are persisted as exact-bytes folded-stacks text (the canonical
//! flamegraph interchange format); the memory-timeline page
//! (the run's `memory.html`) and the summary's `heap` section are the
//! renderings.

use std::path::Path;
use std::sync::OnceLock;

use allocative::{Allocative, FlameGraphBuilder};

use crate::units::{format_memory_size, BYTES_PER_GIB};

/// Output-path prefix for the prover's mid-stage heap snapshots
/// (`{prefix}{label}.folded`). Unset means the harness did not opt in and
/// the prover's cfg-gated emission hooks stay inert — the same
/// [`PPROF_PREFIX`](crate::setup) pattern.
static FLAMEGRAPH_PREFIX: OnceLock<String> = OnceLock::new();

/// Opt in to per-stage flamegraph emission (first call wins).
pub fn set_flamegraph_prefix(prefix: impl Into<String>) {
    let _ = FLAMEGRAPH_PREFIX.set(prefix.into());
}

/// The configured prefix, if the harness opted in.
pub fn flamegraph_prefix() -> Option<&'static str> {
    FLAMEGRAPH_PREFIX.get().map(String::as_str)
}

/// Logs the heap allocation size of an `Allocative`-instrumented value.
pub fn print_data_structure_heap_usage<T: Allocative>(label: &str, data: &T) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let memory_gib = allocative::size_of_unique_allocated_data(data) as f64 / BYTES_PER_GIB;
        tracing::debug!(
            label = label,
            usage = %format_memory_size(memory_gib),
            "heap allocation size"
        );
    }
}

/// Writes a [`FlameGraphBuilder`]'s stacks as folded text (`root;child
/// BYTES` per line, exact bytes): the single persisted form of a heap
/// snapshot, consumed by the summary's `heap` section and the
/// memory-timeline page's icicles.
///
/// Logs a warning and returns on I/O failure instead of panicking.
pub fn write_flamegraph_folded<P: AsRef<Path>>(flamegraph: FlameGraphBuilder, path: P) {
    let folded = flamegraph.finish_and_write_flame_graph();
    if let Err(e) = std::fs::write(path.as_ref(), &folded) {
        tracing::warn!(
            path = %path.as_ref().display(),
            error = %e,
            "failed to write folded heap snapshot"
        );
    }
}
