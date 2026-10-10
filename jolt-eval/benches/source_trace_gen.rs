use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use jolt_eval::objective::performance::source_trace_gen::SourceTraceGenObjective;
use jolt_eval::Objective as _;

fn bench(c: &mut Criterion) {
    std::env::remove_var("TRACER_PARALLEL");
    std::env::remove_var("JOLT_BACKTRACE");
    std::env::remove_var("JOLT_TRACER_CAPACITY_ROWS");
    std::env::set_var("RAYON_NUM_THREADS", "1");
    let objective = SourceTraceGenObjective;
    for setup in objective.setup() {
        let mut group = c.benchmark_group(format!("{}/{}", objective.name(), setup.label));
        group.sample_size(10);
        group.sampling_mode(SamplingMode::Flat);
        group.measurement_time(Duration::from_secs(10));
        group.throughput(Throughput::Elements(setup.row_count as u64));
        let rows = objective.run_source(&setup);
        assert_eq!(rows.rows().len(), setup.row_count);
        drop(rows);
        group.bench_function("source", |b| {
            b.iter(|| std::hint::black_box(objective.run_source(&setup)))
        });
        group.bench_function("reference", |b| {
            b.iter(|| std::hint::black_box(objective.run_reference(&setup)))
        });
        let rows = objective.run_source(&setup);
        group.bench_function("scan", |b| {
            b.iter(|| objective.run_scan(std::hint::black_box(rows.rows())))
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
