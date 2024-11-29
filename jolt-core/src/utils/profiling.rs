#[cfg(not(target_arch = "wasm32"))]
use memory_stats::memory_stats;
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
    println!("================ MEMORY USAGE REPORT ================");

    let memory_usage_map = MEMORY_USAGE_MAP.lock().unwrap();
    for label in memory_usage_map.keys() {
        eprintln!("  Unclosed memory tracing span: \"{}\"", label);
    }

    let memory_delta_map = MEMORY_DELTA_MAP.lock().unwrap();
    for (label, delta) in memory_delta_map.iter() {
        if *delta >= 1.0 {
            println!("  \"{}\": {:.2} GB", label, delta);
        } else {
            println!("  \"{}\": {:.2} MB", label, delta * 1000.0);
        }
    }

    println!("=====================================================");
}

#[cfg(not(target_arch = "wasm32"))]
pub fn print_current_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        let memory_usage_gb = usage.physical_mem as f64 / 1_000_000_000.0;
        if memory_usage_gb >= 1.0 {
            println!("\"{}\" current memory usage: {} GB", label, memory_usage_gb);
        } else {
            println!(
                "\"{}\" current memory usage: {} MB",
                label,
                memory_usage_gb * 1000.0
            );
        }
    } else {
        println!("Failed to get current memory usage (\"{}\")", label);
    }
}
