use memory_stats::memory_stats;

pub fn print_memory_usage(label: &str) {
    if let Some(usage) = memory_stats() {
        println!(
            "<{}> current memory usage: {} GB",
            label,
            usage.physical_mem as f64 / 1_000_000_000.0
        );
    } else {
        println!("Couldn't get the current memory usage :(");
    }
}
