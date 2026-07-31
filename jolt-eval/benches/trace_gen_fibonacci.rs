use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use jolt_eval::guests::Fibonacci;
use jolt_eval::objective::performance::trace_gen::TraceGenObjective;
use jolt_eval::Objective as _;

// Guest input pinned to the `e2e_profiling.rs` default (n = 400000).
//
// Setup (guest build, ELF decode + expansion, one un-timed trace for the row
// count) runs once outside the measurement; each iteration re-traces from the
// shared `JoltProgram`. Backend variants (`x86`, `x86_fast`) join as
// additional bench ids in this group.
fn bench(c: &mut Criterion) {
    // The tracer env-dispatches to parallel mode; pin serial so
    // measurements are environment-independent.
    std::env::remove_var("TRACER_PARALLEL");
    let objective = TraceGenObjective::new(Fibonacci(400000));
    let setup = objective.setup();
    let mut group = c.benchmark_group(objective.name());
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(60));
    group.throughput(Throughput::Elements(setup.trace_len as u64));
    group.bench_function("reference", |b| b.iter(|| objective.run_reference(&setup)));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
