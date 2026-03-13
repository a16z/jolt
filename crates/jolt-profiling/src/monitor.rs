//! Background system metrics monitor.
//!
//! Spawns a thread that periodically samples CPU usage, memory, active cores,
//! and thread count. Metrics are emitted as `tracing::debug!` events with
//! structured `counters.*` fields, compatible with the Perfetto postprocessing
//! script (`scripts/postprocess_trace.py`).

use memory_stats::memory_stats;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use sysinfo::System;

/// Background monitor that samples system metrics at a fixed interval.
///
/// Drop the monitor (or call [`stop`](MetricsMonitor::stop)) to terminate
/// the background thread. The destructor signals the thread and joins it.
pub struct MetricsMonitor {
    handle: Option<JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
}

impl MetricsMonitor {
    /// Starts the monitor with the given sampling interval (in seconds).
    ///
    /// Spawns a background thread named `"metrics-monitor"` that logs:
    /// - `counters.memory_gb` — physical memory usage
    /// - `counters.cpu_percent` — global CPU utilization
    /// - `counters.cores_active_avg` — average active cores
    /// - `counters.cores_active` — cores with >0.1% usage
    /// - `counters.thread_count` — active thread count (Linux only)
    pub fn start(interval_secs: f64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        let handle = thread::Builder::new()
            .name("metrics-monitor".to_string())
            .spawn(move || {
                let interval = Duration::from_millis((interval_secs * 1000.0) as u64);
                let mut system = System::new_all();

                thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);

                while !stop_flag_clone.load(Ordering::Relaxed) {
                    system.refresh_all();

                    let memory_gb = memory_stats().unwrap().physical_mem as f64 / 1_073_741_824.0;
                    let cpu_percent = system.global_cpu_usage();
                    let cores_active_avg = cpu_percent / 100.0 * (system.cpus().len() as f32);
                    let active_cores = system
                        .cpus()
                        .iter()
                        .filter(|cpu| cpu.cpu_usage() > 0.1)
                        .count();

                    let active_threads = std::fs::read_dir("/proc/self/task")
                        .map(|entries| entries.count())
                        .unwrap_or(0);

                    tracing::debug!(
                        counters.memory_gb = memory_gb,
                        counters.cpu_percent = cpu_percent,
                        counters.cores_active_avg = cores_active_avg,
                        counters.cores_active = active_cores,
                        counters.thread_count = active_threads,
                    );

                    thread::sleep(interval);
                }

                tracing::info!("MetricsMonitor stopping");
            })
            .expect("Failed to spawn metrics monitor thread");

        MetricsMonitor {
            handle: Some(handle),
            stop_flag,
        }
    }

    /// Signals the background thread to stop and blocks until it exits.
    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for MetricsMonitor {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
