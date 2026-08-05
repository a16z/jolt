//! Stage-6b IncClaimReduction prepare objective: full four-table setup,
//! including the two raw increment materializations and round ping-pongs.

#![expect(
    clippy::expect_used,
    reason = "benchmark harness must fail loudly on device errors"
)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;
    use std::time::Duration;

    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_kernels::metal::testing::gpu_lock;
    use jolt_kernels::metal::IncPrepareFixture;

    pub fn benchmark(c: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        let mut group = c.benchmark_group("inc_prepare");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        for log_t in [22, 24] {
            let fixture = IncPrepareFixture::production_geometry(log_t);
            fixture.assert_oracle().expect("four-table prepare oracle");
            black_box(fixture.metal_prepare().expect("warm Metal prepare"));
            group.throughput(Throughput::Elements(1u64 << log_t));
            for (label, metal) in [
                ("host_a", false),
                ("metal_a", true),
                ("metal_b", true),
                ("host_b", false),
            ] {
                group.bench_with_input(BenchmarkId::new(label, log_t), &metal, |bencher, metal| {
                    bencher.iter(|| {
                        let prepared = if *metal {
                            fixture.metal_prepare().expect("Metal prepare")
                        } else {
                            fixture.host_prepare().expect("host prepare")
                        };
                        black_box(prepared)
                    });
                });
            }
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
    eprintln!("inc_prepare requires macOS");
}
