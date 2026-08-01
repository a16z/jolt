//! Heap flamegraph generation from `allocative`-instrumented data structures.

use std::sync::OnceLock;
use std::{fs::File, io::Cursor, path::Path};

use allocative::{Allocative, FlameGraphBuilder};
use inferno::flamegraph::Options;

use crate::units::{format_memory_size, BYTES_PER_GIB, BYTES_PER_MIB};

/// Output-path prefix for the prover's per-stage flamegraph SVGs
/// (`{prefix}stage{N}.svg`). Unset means the harness did not opt in and the
/// prover's cfg-gated emission hooks stay inert — the same
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

/// Renders a [`FlameGraphBuilder`] to an SVG flamegraph file.
///
/// Uses `inferno` for rendering with MiB units and flame-chart mode.
/// Logs a warning and returns on I/O failure instead of panicking.
pub fn write_flamegraph_svg<P: AsRef<Path>>(flamegraph: FlameGraphBuilder, path: P) {
    let mut opts = Options::default();
    opts.color_diffusion = true;
    opts.count_name = String::from("MiB");
    opts.factor = 1.0 / BYTES_PER_MIB;
    opts.flame_chart = true;

    let flamegraph_src = flamegraph.finish_and_write_flame_graph();
    // The machine-queryable twin: exact byte counts in the canonical
    // folded-stacks format ("root;child BYTES" per line). The SVG's hover
    // text is integer-MiB rounded; queries and the summary's heap section
    // read this instead.
    let folded_path = path.as_ref().with_extension("folded");
    if let Err(e) = std::fs::write(&folded_path, &flamegraph_src) {
        tracing::warn!(
            path = %folded_path.display(),
            error = %e,
            "failed to write folded flamegraph"
        );
    }
    let input = Cursor::new(flamegraph_src);

    let output = match File::create(path.as_ref()) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                path = %path.as_ref().display(),
                error = %e,
                "failed to create flamegraph SVG file"
            );
            return;
        }
    };

    if let Err(e) = inferno::flamegraph::from_reader(&mut opts, input, output) {
        tracing::warn!(
            path = %path.as_ref().display(),
            error = %e,
            "failed to render flamegraph SVG"
        );
    }
}
