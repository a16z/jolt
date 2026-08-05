//! Stage-3 instruction-input round-0 isolated objective.
//!
//! Deterministic native trace lanes and production split-eq geometry; setup,
//! device buffers, oracle, and warm dispatch stay outside Criterion timing.

#![expect(
    clippy::expect_used,
    reason = "benchmark harness must fail loudly on device errors"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::InstructionInputRound0Fixture;

    pub fn benchmark(c: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        if std::env::var_os("JOLT_INSTR_INPUT_ATTRIBUTION_ONCE").is_some() {
            for log_t in [22, 24] {
                let fixture = InstructionInputRound0Fixture::production_geometry(log_t)
                    .expect("round-0 fixture");
                fixture.assert_oracle().expect("round-0 oracle");
                let start = Instant::now();
                black_box(fixture.host_message());
                eprintln!(
                    "[instruction-input] 2^{log_t} host_q0={:?}",
                    start.elapsed()
                );
                let start = Instant::now();
                black_box(fixture.metal_message().expect("q0 dispatch"));
                eprintln!(
                    "[instruction-input] 2^{log_t} metal_q0={:?}",
                    start.elapsed()
                );
                let start = Instant::now();
                fixture.dense_bind().expect("bind-native dispatch");
                eprintln!(
                    "[instruction-input] 2^{log_t} dense_bind={:?}",
                    start.elapsed()
                );
            }
            return;
        }

        let mut group = c.benchmark_group("instruction_input_round0");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        for log_t in [22, 24] {
            let fixture =
                InstructionInputRound0Fixture::production_geometry(log_t).expect("round-0 fixture");
            fixture.assert_oracle().expect("round-0 oracle");
            fixture.dense_bind().expect("warm bind-native dispatch");
            group.throughput(Throughput::Elements(1u64 << log_t));
            group.bench_with_input(BenchmarkId::new("host_message", log_t), &log_t, |b, _| {
                b.iter(|| black_box(fixture.host_message()));
            });
            group.bench_with_input(BenchmarkId::new("metal_message", log_t), &log_t, |b, _| {
                b.iter(|| black_box(fixture.metal_message().expect("q0 dispatch")));
            });
            group.bench_with_input(BenchmarkId::new("dense_bind", log_t), &log_t, |b, _| {
                b.iter(|| fixture.dense_bind().expect("bind-native dispatch"));
            });
        }
        group.finish();
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(benches, macos::benchmark);
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("instruction_input_round0 requires macOS");
}
