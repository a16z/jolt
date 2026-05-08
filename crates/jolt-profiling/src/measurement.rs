//! Small measurement primitives for perf gates and benchmark harnesses.
//!
//! These helpers are intentionally generic. Protocol-specific harnesses should
//! live near the protocol or test that owns them, while timing, medians, and
//! peak RSS sampling stay here so core-vs-Bolt gates report comparable metrics.

use std::time::Instant;

/// Time a closure, returning elapsed milliseconds and the closure result.
pub fn time_it<T>(f: impl FnOnce() -> T) -> (f64, T) {
    let start = Instant::now();
    let value = f();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    (elapsed_ms, value)
}

/// Median of a non-empty finite `f64` slice.
///
/// Returns `None` for empty input or if any value is NaN/infinite. Perf gates
/// should treat `None` as invalid measurement input rather than silently sorting
/// a partial order.
pub fn median_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let len = sorted.len();
    Some(if len % 2 == 1 {
        sorted[len / 2]
    } else {
        f64::midpoint(sorted[len / 2 - 1], sorted[len / 2])
    })
}

/// Median of a non-empty `u64` slice.
pub fn median_u64(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let len = sorted.len();
    Some(if len % 2 == 1 {
        sorted[len / 2]
    } else {
        u64::midpoint(sorted[len / 2 - 1], sorted[len / 2])
    })
}

#[cfg(not(target_arch = "wasm32"))]
mod peak_rss {
    use std::io;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    use memory_stats::memory_stats;

    /// Peak resident-set-size sampler running on a background thread.
    ///
    /// Samples physical memory at a fixed cadence. Stopping the sampler drains
    /// the thread and returns the peak observed physical memory in megabytes
    /// using 1 MB = 1,000,000 bytes, matching `memory-stats` reporting.
    #[must_use = "call finish() to stop the sampler and read the peak RSS"]
    pub struct PeakRssSampler {
        peak_bytes: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl PeakRssSampler {
        /// Start sampling RSS.
        pub fn start() -> io::Result<Self> {
            let peak_bytes = Arc::new(AtomicU64::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let peak_clone = Arc::clone(&peak_bytes);
            let stop_clone = Arc::clone(&stop);

            sample_peak(&peak_clone);

            let handle = thread::Builder::new()
                .name("jolt-profiling-rss".into())
                .spawn(move || {
                    while !stop_clone.load(Ordering::Relaxed) {
                        sample_peak(&peak_clone);
                        thread::sleep(Duration::from_millis(50));
                    }
                })?;

            Ok(Self {
                peak_bytes,
                stop,
                handle: Some(handle),
            })
        }

        /// Stop the sampler, join the thread, and return peak RSS in MB.
        pub fn finish(mut self) -> u64 {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            sample_peak(&self.peak_bytes);
            self.peak_bytes.load(Ordering::Relaxed) / 1_000_000
        }
    }

    fn sample_peak(peak_bytes: &AtomicU64) {
        if let Some(stats) = memory_stats() {
            let _ = peak_bytes.fetch_max(stats.physical_mem as u64, Ordering::Relaxed);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use peak_rss::PeakRssSampler;

#[cfg(test)]
mod tests {
    use super::{median_f64, median_u64};

    #[test]
    fn median_f64_rejects_invalid_input() {
        assert_eq!(median_f64(&[]), None);
        assert_eq!(median_f64(&[1.0, f64::NAN]), None);
    }

    #[test]
    fn median_f64_handles_odd_and_even_lengths() {
        assert_eq!(median_f64(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median_f64(&[10.0, 2.0, 4.0, 8.0]), Some(6.0));
    }

    #[test]
    fn median_u64_handles_odd_and_even_lengths() {
        assert_eq!(median_u64(&[]), None);
        assert_eq!(median_u64(&[3, 1, 2]), Some(2));
        assert_eq!(median_u64(&[10, 2, 4, 8]), Some(6));
    }
}
