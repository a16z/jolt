//! Heap flamegraph generation from `allocative`-instrumented data structures.

use allocative::{Allocative, FlameGraphBuilder};
use std::path::Path;

/// Logs the heap allocation size of an `Allocative`-instrumented value.
pub fn print_data_structure_heap_usage<T: Allocative>(label: &str, data: &T) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let memory_usage_gb =
            allocative::size_of_unique_allocated_data(data) as f64 / 1_000_000_000.0;
        if memory_usage_gb >= 1.0 {
            tracing::debug!("\"{label}\" memory usage: {memory_usage_gb:.2} GB");
        } else {
            tracing::debug!(
                "\"{}\" memory usage: {:.2} MB",
                label,
                memory_usage_gb * 1000.0
            );
        }
    }
}

/// Renders a [`FlameGraphBuilder`] to an SVG flamegraph file.
///
/// Uses `inferno` for rendering with MB units and flame-chart mode.
pub fn write_flamegraph_svg<P: AsRef<Path>>(flamegraph: FlameGraphBuilder, path: P) {
    use std::{fs::File, io::Cursor};

    use inferno::flamegraph::Options;

    let mut opts = Options::default();
    opts.color_diffusion = true;
    opts.count_name = String::from("MB");
    opts.factor = 0.000001;
    opts.flame_chart = true;

    let flamegraph_src = flamegraph.finish_and_write_flame_graph();
    let input = Cursor::new(flamegraph_src);
    let output = File::create(path).unwrap();
    inferno::flamegraph::from_reader(&mut opts, input, output).unwrap();
}
