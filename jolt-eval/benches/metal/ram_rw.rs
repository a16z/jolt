//! Stage-2 RAM read/write CSR-build and device-prefix objectives.

#![expect(
    clippy::print_stdout,
    reason = "benchmark harness must fail loudly and report its schedule"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Duration;

    use criterion::{BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput};
    use jolt_kernels::metal::ram_rw_bench::{
        assert_small_scale_prepare_parity, assert_small_scale_round_parity, BenchConfig,
        PrepareFixture, RoundFixture,
    };
    use jolt_kernels::metal::testing::gpu_lock;

    pub fn benchmark(criterion: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        assert_small_scale_prepare_parity();
        assert_small_scale_round_parity();
        println!("ram_rw oracle: GPU CSR byte-equal; legacy/fused wire-equal to CPU twin");

        for log_t in [22, 24] {
            {
                let fixture = Arc::new(PrepareFixture::synthetic(BenchConfig::production(log_t)));
                let mut group = criterion.benchmark_group("ram_rw_prepare_csr");
                group.sample_size(10);
                group.sampling_mode(SamplingMode::Flat);
                group.warm_up_time(Duration::from_millis(100));
                group.measurement_time(Duration::from_secs(2));
                group.throughput(Throughput::Elements(fixture.cycles() as u64));
                for (label, gpu) in [
                    ("serial_a", false),
                    ("gpu_a", true),
                    ("gpu_b", true),
                    ("serial_b", false),
                ] {
                    group.bench_with_input(
                        BenchmarkId::new(label, format!("2^{log_t}")),
                        &gpu,
                        |bencher, &gpu| {
                            bencher.iter_custom(|iterations| {
                                let mut measured = Duration::ZERO;
                                for _ in 0..iterations {
                                    let timing = fixture.run(gpu);
                                    assert_eq!(timing.command_buffers, u64::from(gpu));
                                    assert_eq!(timing.kernel_dispatches, u64::from(gpu));
                                    black_box(timing.entry_bytes);
                                    measured += timing.total;
                                }
                                measured
                            });
                        },
                    );
                }
                group.finish();
            }

            {
                let fixture = Arc::new(RoundFixture::synthetic(BenchConfig::production(log_t)));
                let mut group = criterion.benchmark_group("ram_rw_prefix");
                group.sample_size(10);
                group.sampling_mode(SamplingMode::Flat);
                group.warm_up_time(Duration::from_millis(100));
                group.measurement_time(Duration::from_secs(2));
                group.throughput(Throughput::Elements(fixture.cycles() as u64));
                for (label, fused) in [
                    ("legacy_a", false),
                    ("fused_a", true),
                    ("fused_b", true),
                    ("legacy_b", false),
                ] {
                    group.bench_with_input(
                        BenchmarkId::new(label, format!("2^{log_t}")),
                        &fused,
                        |bencher, &fused| {
                            bencher.iter_batched_ref(
                                || fixture.prepare(fused),
                                |rounds| {
                                    let timing = rounds.run();
                                    black_box((
                                        timing.total,
                                        timing.command_buffers,
                                        timing.kernel_dispatches,
                                    ))
                                },
                                BatchSize::LargeInput,
                            );
                        },
                    );
                }
                group.finish();
            }
        }
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, macos::benchmark);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ram_rw requires macOS");
}
