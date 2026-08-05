//! Stage-4 registers read/write sparse-prefix schedule benchmark.

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
    use jolt_kernels::metal::st4_registers_rw_bench::{
        assert_small_scale_parity, assert_small_scale_prepare_parity, BenchConfig, BenchFixture,
        PrepareBenchFixture,
    };
    use jolt_kernels::metal::testing::gpu_lock;

    pub fn benchmark(criterion: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        assert_small_scale_prepare_parity();
        assert_small_scale_parity();
        println!(
            "registers_rw oracle: GPU prepare byte-equal to serial CSR; legacy/fused wire-equal to CPU twin"
        );

        for log_t in [22, 24] {
            let prepare_fixture = Arc::new(PrepareBenchFixture::synthetic(
                BenchConfig::production(log_t),
            ));
            let mut prepare_group = criterion.benchmark_group("registers_rw_prepare");
            prepare_group.sample_size(10);
            prepare_group.sampling_mode(SamplingMode::Flat);
            prepare_group.warm_up_time(Duration::from_millis(100));
            prepare_group.measurement_time(Duration::from_secs(2));
            prepare_group.throughput(Throughput::Elements(prepare_fixture.cycles() as u64));
            for (label, gpu) in [("serial", false), ("gpu", true)] {
                prepare_group.bench_with_input(
                    BenchmarkId::new(label, format!("2^{log_t}")),
                    &gpu,
                    |bencher, &target_gpu| {
                        bencher.iter_custom(|iterations| {
                            let mut measured = Duration::ZERO;
                            for _ in 0..iterations {
                                black_box(prepare_fixture.run(!target_gpu));
                                let timing = prepare_fixture.run(target_gpu);
                                assert_eq!(timing.command_buffers, u64::from(target_gpu));
                                assert_eq!(timing.kernel_dispatches, u64::from(target_gpu));
                                black_box(timing.entry_bytes);
                                measured += timing.total;
                            }
                            measured
                        });
                    },
                );
            }
            prepare_group.finish();

            let fixture = Arc::new(BenchFixture::synthetic(BenchConfig::production(log_t)));
            let mut group = criterion.benchmark_group("registers_rw_prefix");
            group.sample_size(10);
            group.sampling_mode(SamplingMode::Flat);
            group.warm_up_time(Duration::from_millis(100));
            group.measurement_time(Duration::from_secs(2));
            group.throughput(Throughput::Elements(fixture.cycles() as u64));
            for (label, fused) in [("legacy", false), ("fused", true)] {
                group.bench_with_input(
                    BenchmarkId::new(label, format!("2^{log_t}")),
                    &fused,
                    |bencher, &target_fused| {
                        bencher.iter_batched_ref(
                            || {
                                let _companion = fixture.run_device_pass(!target_fused);
                                fixture.prepare_device_pass(target_fused)
                            },
                            |pass| black_box(pass.run()),
                            BatchSize::LargeInput,
                        );
                    },
                );
            }
            group.finish();
        }
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, macos::benchmark);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("registers_rw_prefix requires macOS");
}
