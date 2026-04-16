use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Peak resident-set-size sampler running on a background thread.
///
/// Samples `memory_stats::memory_stats()` at ~50 ms cadence. Stopping the
/// sampler drains the thread and returns the peak observed `physical_mem`
/// in megabytes (1 MB = 1_000_000 bytes — chosen to match `memory-stats`'
/// own reporting and what users expect to see in `ps`/`top`).
pub struct PeakRssSampler {
    peak_bytes: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl PeakRssSampler {
    pub fn start() -> Self {
        let peak_bytes = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let peak_clone = peak_bytes.clone();
        let stop_clone = stop.clone();

        // Sample once synchronously so the baseline is captured even if the
        // prove call returns before the background thread's first tick.
        if let Some(stats) = memory_stats::memory_stats() {
            let _ = peak_clone.fetch_max(stats.physical_mem as u64, Ordering::Relaxed);
        }

        let handle = thread::Builder::new()
            .name("jolt-bench-rss".into())
            .spawn(move || {
                while !stop_clone.load(Ordering::Relaxed) {
                    if let Some(stats) = memory_stats::memory_stats() {
                        let _ = peak_clone.fetch_max(stats.physical_mem as u64, Ordering::Relaxed);
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            })
            .expect("spawn rss sampler thread");

        Self {
            peak_bytes,
            stop,
            handle: Some(handle),
        }
    }

    /// Stop the sampler, join the thread, return peak RSS in MB (1_000_000 bytes).
    pub fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        if let Some(stats) = memory_stats::memory_stats() {
            let _ = self
                .peak_bytes
                .fetch_max(stats.physical_mem as u64, Ordering::Relaxed);
        }
        self.peak_bytes.load(Ordering::Relaxed) / 1_000_000
    }
}

/// Time a closure, returning (elapsed_ms, result).
pub fn time_it<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let start = Instant::now();
    let value = f();
    let elapsed = start.elapsed();
    let ms = (elapsed.as_nanos() as f64) / 1_000_000.0;
    (ms, value)
}

/// Median of a non-empty slice. Panics if `values` is empty.
pub fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "median of empty slice");
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    }
}

/// Integer median. Rounds to nearest for even-length input.
pub fn median_u64(values: &[u64]) -> u64 {
    assert!(!values.is_empty(), "median of empty slice");
    let mut sorted: Vec<u64> = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        u64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    }
}
