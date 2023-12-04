use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};

use liblasso::sumcheck_bench;

fn criterion_config(
    c: &mut Criterion,
) -> criterion::BenchmarkGroup<'_, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group("sumcheck");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    group
}

fn bench(c: &mut Criterion) {
    let mut group = criterion_config(c);
    sumcheck_bench(&mut group);
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
