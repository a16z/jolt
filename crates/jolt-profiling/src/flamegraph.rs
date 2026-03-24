//! Heap flamegraph generation from `allocative`-instrumented data structures.

use std::{fs::File, io::Cursor, path::Path};

use allocative::{Allocative, FlameGraphBuilder};
use inferno::flamegraph::Options;

use crate::units::{format_memory_size, BYTES_PER_GIB};

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
    opts.factor = 1.0 / BYTES_PER_GIB * 1024.0;
    opts.flame_chart = true;

    let flamegraph_src = flamegraph.finish_and_write_flame_graph();
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
