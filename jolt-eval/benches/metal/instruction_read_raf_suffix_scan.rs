//! Stage-5 InstructionReadRAF suffix-scan isolated objective.
//!
//! Production scanner schedule at 2^22/2^24; setup, exact CPU parity, and
//! buffer wrapping stay outside timing.

#![expect(
    clippy::expect_used,
    clippy::print_stderr,
    reason = "benchmark harness must fail loudly on device errors"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::time::{Duration, Instant};

    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::IrrSuffixScanFixture;

    pub fn benchmark(c: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        if let Some(target) = std::env::var("JOLT_IRR_SUFFIX_PROBE_SGS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            let log_t = std::env::var("JOLT_IRR_SUFFIX_PROBE_LOG_T")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(22);
            let mut fixture = IrrSuffixScanFixture::with_simdgroups(log_t, target)
                .expect("suffix-scan probe fixture");
            fixture.assert_oracle().expect("suffix-scan CPU parity");
            let buffers = fixture.buffers().expect("suffix-scan buffers");
            buffers.run().expect("warm optimized dispatch");
            buffers
                .run_with_reduce()
                .expect("warm scan+reduce dispatch");
            let scan_samples: Vec<_> = (0..30).map(|_| elapsed_ms(|| buffers.run())).collect();
            let combined_samples: Vec<_> = (0..30)
                .map(|_| elapsed_ms(|| buffers.run_with_reduce()))
                .collect();
            let (scan_mean, scan_half) = mean_ci95(&scan_samples);
            let (combined_mean, combined_half) = mean_ci95(&combined_samples);
            eprintln!(
                "[irr-suffix-occupancy] simdgroups={target} 2^{log_t} scan={scan_mean:.3}±{scan_half:.3}ms combined={combined_mean:.3}±{combined_half:.3}ms"
            );
            return;
        }
        if std::env::var_os("JOLT_IRR_RETENTION_ONCE").is_some() {
            for log_t in [22, 24] {
                let mut fixture =
                    IrrSuffixScanFixture::production_geometry(log_t).expect("suffix-scan fixture");
                fixture.assert_oracle().expect("suffix-scan CPU parity");
                let buffers = fixture.buffers().expect("suffix-scan buffers");
                buffers.run_legacy().expect("warm legacy dispatch");
                buffers.run().expect("warm optimized dispatch");
                let mut legacy = Vec::with_capacity(30);
                let mut optimized = Vec::with_capacity(30);
                for sample in 0..30 {
                    if sample % 2 == 0 {
                        legacy.push(elapsed_ms(|| buffers.run_legacy()));
                        optimized.push(elapsed_ms(|| buffers.run()));
                    } else {
                        optimized.push(elapsed_ms(|| buffers.run()));
                        legacy.push(elapsed_ms(|| buffers.run_legacy()));
                    }
                }
                let (legacy_mean, legacy_half) = mean_ci95(&legacy);
                let (optimized_mean, optimized_half) = mean_ci95(&optimized);
                eprintln!(
                    "[irr-suffix-retention] 2^{log_t} legacy={legacy_mean:.3}±{legacy_half:.3}ms optimized={optimized_mean:.3}±{optimized_half:.3}ms speedup={:.1}%",
                    100.0 * (legacy_mean - optimized_mean) / legacy_mean
                );
            }
            return;
        }

        let mut group = c.benchmark_group("instruction_read_raf_suffix_scan");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        for log_t in [22, 24] {
            let mut fixture =
                IrrSuffixScanFixture::production_geometry(log_t).expect("suffix-scan fixture");
            fixture.assert_oracle().expect("suffix-scan CPU parity");
            let buffers = fixture.buffers().expect("suffix-scan buffers");
            buffers.run_legacy().expect("warm legacy dispatch");
            buffers.run().expect("warm optimized dispatch");
            group.throughput(Throughput::Elements(1u64 << log_t));
            group.bench_with_input(BenchmarkId::new("legacy", log_t), &log_t, |b, _| {
                b.iter(|| buffers.run_legacy().expect("legacy suffix-scan dispatch"));
            });
            group.bench_with_input(BenchmarkId::new("collision_only", log_t), &log_t, |b, _| {
                b.iter(|| buffers.run().expect("suffix-scan dispatch"));
            });
        }
        group.finish();
    }

    fn elapsed_ms(run: impl FnOnce() -> Result<(), jolt_kernels::metal::MetalError>) -> f64 {
        let start = Instant::now();
        run().expect("suffix-scan dispatch");
        start.elapsed().as_secs_f64() * 1e3
    }

    fn mean_ci95(samples: &[f64]) -> (f64, f64) {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        (mean, 2.045 * (variance / n).sqrt())
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, macos::benchmark);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("instruction_read_raf_suffix_scan requires macOS");
}
