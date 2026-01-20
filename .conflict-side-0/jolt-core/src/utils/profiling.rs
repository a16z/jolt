#[cfg(feature = "allocative")]
use allocative::{Allocative, FlameGraphBuilder};
#[cfg(not(target_arch = "wasm32"))]
use memory_stats::memory_stats;
#[cfg(feature = "allocative")]
use std::path::Path;
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
static MEMORY_USAGE_MAP: LazyLock<Mutex<HashMap<&'static str, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static MEMORY_DELTA_MAP: LazyLock<Mutex<HashMap<&'static str, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(not(target_arch = "wasm32"))]
pub fn start_memory_tracing_span(label: &'static str) {
    let memory_usage = memory_stats().unwrap().physical_mem;
    let mut map = MEMORY_USAGE_MAP.lock().unwrap();
    assert_eq!(
        map.insert(label, memory_usage as f64 / 1_000_000_000.0),
        None
    );
}

#[cfg(not(target_arch = "wasm32"))]
pub fn end_memory_tracing_span(label: &'static str) {
    let memory_usage_end = memory_stats().unwrap().physical_mem as f64 / 1_000_000_000.0;
    let mut memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    let memory_usage_start = memory_usage_map.remove(label).unwrap();

    let memory_usage_delta = memory_usage_end - memory_usage_start;
    let mut memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
    assert_eq!(memory_delta_map.insert(label, memory_usage_delta), None);
}

pub fn report_memory_usage() {
    tracing::info!("================ MEMORY USAGE REPORT ================");

    let memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    for label in memory_usage_map.keys() {
        tracing::warn!("  Unclosed memory tracing span: \"{label}\"");
    }

    let memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
    for (label, delta) in memory_delta_map.iter() {
        if *delta >= 1.0 {
            tracing::info!("  \"{label}\": {delta:.2} GB");
        } else {
            tracing::info!("  \"{}\": {:.2} MB", label, delta * 1000.0);
        }
    }

    tracing::info!("=====================================================");
}

#[cfg(not(target_arch = "wasm32"))]
pub fn print_current_memory_usage(label: &str) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        if let Some(usage) = memory_stats() {
            let memory_usage_gb = usage.physical_mem as f64 / 1_000_000_000.0;
            if memory_usage_gb >= 1.0 {
                tracing::debug!("\"{label}\" current memory usage: {memory_usage_gb:.2} GB");
            } else {
                tracing::debug!(
                    "\"{}\" current memory usage: {:.2} MB",
                    label,
                    memory_usage_gb * 1000.0
                );
            }
        } else {
            tracing::debug!("Failed to get current memory usage (\"{label}\")");
        }
    }
}

#[cfg(feature = "allocative")]
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

#[cfg(feature = "allocative")]
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
