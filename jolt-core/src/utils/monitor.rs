use memory_stats::memory_stats;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use sysinfo::System;

pub struct MetricsMonitor {
    handle: Option<JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
}

impl MetricsMonitor {
    pub fn start(interval_secs: f64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();

        let handle = thread::Builder::new()
            .name("metrics-monitor".to_string())
            .spawn(move || {
                let interval = Duration::from_millis((interval_secs * 1000.0) as u64);

                let mut system = System::new_all();

                thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);

                // TODO: Spawn a thread to monitor intel pcm for memory bandwidth

                while !stop_flag_clone.load(Ordering::Relaxed) {
                    system.refresh_all();

                    // Collect metrics
                    let memory_gb = memory_stats().unwrap().physical_mem as f64 / 1_073_741_824.0;
                    let cpu_percent = system.global_cpu_usage();
                    let cores_active_avg = cpu_percent / 100.0 * (system.cpus().len() as f32);
                    // Report number of cores with CPU usage > 0.1%
                    let active_cores_5per = system
                        .cpus()
                        .iter()
                        .filter(|cpu| cpu.cpu_usage() > 0.1)
                        .count();

                    // Get current process info for thread counting via /proc filesystem
                    let active_threads = std::fs::read_dir("/proc/self/task")
                        .map(|entries| entries.count())
                        .unwrap_or(0);

                    // Log with clear prefix for filtering
                    tracing::debug!(
                        counters.memory_gb = memory_gb,
                        counters.cpu_percent = cpu_percent,
                        counters.cores_active_avg = cores_active_avg,
                        counters.cores_active = active_cores_5per,
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

    pub fn stop(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for MetricsMonitor {
    fn drop(&mut self) {
        // Signal the monitoring thread to stop
        self.stop_flag.store(true, Ordering::Relaxed);

        // Wait for the thread to finish
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
