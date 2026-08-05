//! Stage-5 InstructionReadRAF phase-scan isolated objective.
//!
//! Production row shape and scanner schedule at 2^22/2^24; setup, exact
//! CPU parity, and buffer wrapping stay outside Criterion timing.

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
    use jolt_kernels::metal::IrrPhaseScanFixture;

    pub fn benchmark(c: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        if std::env::var_os("JOLT_IRR_RETENTION_ONCE").is_some() {
            for log_t in [22, 24] {
                let mut fixture =
                    IrrPhaseScanFixture::production_geometry(log_t).expect("phase-scan fixture");
                fixture.assert_oracle().expect("phase-scan CPU parity");
                let buffers = fixture.buffers().expect("phase-scan buffers");
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
                    "[irr-retention] 2^{log_t} legacy={legacy_mean:.3}±{legacy_half:.3}ms optimized={optimized_mean:.3}±{optimized_half:.3}ms speedup={:.1}%",
                    100.0 * (legacy_mean - optimized_mean) / legacy_mean
                );
            }
            return;
        }
        let mut group = c.benchmark_group("instruction_read_raf_phase_scan");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        for log_t in [22, 24] {
            let mut fixture =
                IrrPhaseScanFixture::production_geometry(log_t).expect("phase-scan fixture");
            fixture.assert_oracle().expect("phase-scan CPU parity");
            let buffers = fixture.buffers().expect("phase-scan buffers");
            buffers
                .run_legacy()
                .expect("warm legacy phase-scan dispatch");
            buffers.run().expect("warm phase-scan dispatch");
            group.throughput(Throughput::Elements(1u64 << log_t));
            group.bench_with_input(BenchmarkId::new("legacy", log_t), &log_t, |b, _| {
                b.iter(|| buffers.run_legacy().expect("legacy phase-scan dispatch"));
            });
            group.bench_with_input(BenchmarkId::new("collision_only", log_t), &log_t, |b, _| {
                b.iter(|| buffers.run().expect("phase-scan dispatch"));
            });
        }
        group.finish();
    }

    fn elapsed_ms(run: impl FnOnce() -> Result<(), jolt_kernels::metal::MetalError>) -> f64 {
        let start = Instant::now();
        run().expect("phase-scan dispatch");
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
    eprintln!("instruction_read_raf_phase_scan requires macOS");
}
