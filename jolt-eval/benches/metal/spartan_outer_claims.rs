//! Stage-1 Spartan-outer final-opening isolated objective.
//!
//! Deterministic trace lanes and production split-eq geometry; fixture,
//! oracle, and warm dispatch stay outside Criterion timing.

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
    use jolt_kernels::metal::SpartanOuterClaimsFixture;

    pub fn benchmark(c: &mut Criterion) {
        let _gpu_lock = gpu_lock();
        if std::env::var_os("JOLT_OUTER_ATTRIBUTION_ONCE").is_some() {
            for log_t in [22, 24] {
                let fixture = SpartanOuterClaimsFixture::production_geometry(log_t)
                    .expect("outer claims fixture");
                fixture.assert_oracle().expect("outer claims oracle");
                eprintln!(
                    "[spartan-outer] 2^{log_t} {:?}",
                    fixture.attribute().expect("attribution")
                );
            }
            return;
        }

        let mut group = c.benchmark_group("spartan_outer_claims");
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(5));
        for log_t in [22, 24] {
            let fixture = SpartanOuterClaimsFixture::production_geometry(log_t)
                .expect("outer claims fixture");
            fixture.assert_oracle().expect("outer claims oracle");
            group.throughput(Throughput::Elements(1u64 << log_t));
            group.bench_with_input(BenchmarkId::new("host", log_t), &log_t, |b, _| {
                b.iter(|| black_box(fixture.host_claims()));
            });
            group.bench_with_input(BenchmarkId::new("metal", log_t), &log_t, |b, _| {
                b.iter(|| black_box(fixture.metal_claims().expect("claims dispatch")));
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
    eprintln!("spartan_outer_claims requires macOS");
}
