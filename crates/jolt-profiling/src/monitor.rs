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

use crate::units::BYTES_PER_GIB;

/// Samples Apple GPU utilization from the IOAccelerator registry entry
/// (`ioreg` subprocess, ~10-15 ms, no privileges required). Returns
/// `(device_util_pct, renderer_util_pct)`; max across accelerators when a
/// machine exposes several.
#[cfg(target_os = "macos")]
fn sample_gpu_percent() -> Option<(f64, f64)> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let find_max = |key: &str| -> Option<f64> {
        let mut best: Option<f64> = None;
        for (pos, _) in text.match_indices(key) {
            let tail = text[pos + key.len()..].trim_start_matches(['"', '=', ' ']);
            let digits: &str = &tail[..tail
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(tail.len())];
            if let Ok(v) = digits.parse::<f64>() {
                best = Some(best.map_or(v, |b: f64| b.max(v)));
            }
        }
        best
    };
    let device = find_max("\"Device Utilization %\"")?;
    let renderer = find_max("\"Renderer Utilization %\"").unwrap_or(0.0);
    Some((device, renderer))
}

#[cfg(not(target_os = "macos"))]
fn sample_gpu_percent() -> Option<(f64, f64)> {
    None
}

/// Background monitor that samples system metrics at a fixed interval.
///
/// Drop the monitor to terminate the background thread. The destructor
/// signals the thread and joins it.
#[must_use = "monitor stops when dropped"]
pub struct MetricsMonitor {
    handle: Option<JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
}

impl MetricsMonitor {
    /// Starts the monitor with the given sampling interval (in seconds).
    ///
    /// Spawns a background thread named `"metrics-monitor"` that logs:
    /// - `counters.memory_gib` — physical memory usage
    /// - `counters.cpu_percent` — global CPU utilization
    /// - `counters.cores_active_avg` — average active cores
    /// - `counters.cores_active` — cores with >0.1% usage
    /// - `counters.thread_count` — active thread count (Linux only, 0 elsewhere)
    pub fn start(interval_secs: f64) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop = stop_flag.clone();

        let spawn_result = thread::Builder::new()
            .name("metrics-monitor".to_string())
            .spawn(move || {
                let interval = Duration::from_millis(((interval_secs * 1000.0) as u64).max(50));
                let mut system = System::new();

                thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);

                while !stop.load(Ordering::Acquire) {
                    system.refresh_cpu_all();

                    let memory_gib =
                        memory_stats().map_or(0.0, |s| s.physical_mem as f64 / BYTES_PER_GIB);
                    let cpu_percent = system.global_cpu_usage();
                    let cores_active_avg = cpu_percent / 100.0 * (system.cpus().len() as f32);
                    let active_cores = system
                        .cpus()
                        .iter()
                        .filter(|cpu| cpu.cpu_usage() > 0.1)
                        .count();

                    #[cfg(target_os = "linux")]
                    let active_threads =
                        std::fs::read_dir("/proc/self/task").map_or(0, |entries| entries.count());

                    #[cfg(not(target_os = "linux"))]
                    let active_threads = 0_usize;

                    let (gpu_percent, gpu_renderer_percent) =
                        sample_gpu_percent().unwrap_or((0.0, 0.0));

                    tracing::debug!(
                        counters.memory_gib = memory_gib,
                        counters.cpu_percent = cpu_percent,
                        counters.cores_active_avg = cores_active_avg,
                        counters.cores_active = active_cores,
                        counters.thread_count = active_threads,
                        counters.gpu_percent = gpu_percent,
                        counters.gpu_renderer_percent = gpu_renderer_percent,
                    );

                    thread::sleep(interval);
                }

                tracing::info!("MetricsMonitor stopping");
            });

        let handle = match spawn_result {
            Ok(h) => Some(h),
            Err(e) => {
                tracing::warn!(error = %e, "failed to spawn metrics monitor thread");
                None
            }
        };

        MetricsMonitor { handle, stop_flag }
    }
}

impl Drop for MetricsMonitor {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
