use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use jolt_eval::guests::Fibonacci;
use jolt_eval::objective::performance::trace_gen::TraceGenObjective;
use jolt_eval::Objective as _;

fn bench(c: &mut Criterion) {
    std::env::remove_var("TRACER_PARALLEL");
    let objective = TraceGenObjective::new(Fibonacci(400000));
    let setup = objective.setup();
    let mut group = c.benchmark_group(objective.name());
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.throughput(Throughput::Elements(setup.trace_len as u64));
    group.bench_function("reference", |b| {
        b.iter(|| assert_eq!(objective.run_reference(&setup), setup.trace_len))
    });
    group.bench_function("reference_raw", |b| {
        b.iter(|| assert_eq!(objective.run_reference_raw(&setup), setup.trace_len))
    });
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        let mut backend = jolt_tracer_x86::X86TracerBackend::new();
        assert_eq!(
            objective.run_x86_fast(&mut backend, &setup),
            setup.trace_len
        );
        group.bench_function("x86", |b| {
            b.iter(|| assert_eq!(objective.run_x86(&mut backend, &setup), setup.trace_len))
        });
        group.bench_function("x86_fast", |b| {
            b.iter(|| {
                assert_eq!(
                    objective.run_x86_fast(&mut backend, &setup),
                    setup.trace_len
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
