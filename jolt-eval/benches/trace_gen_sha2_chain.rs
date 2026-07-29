use criterion::{criterion_group, criterion_main, Criterion, SamplingMode, Throughput};
use jolt_eval::guests::Sha2Chain;
use jolt_eval::objective::performance::trace_gen::TraceGenObjective;
use jolt_eval::Objective as _;

// Ensure the SHA2 inline library is linked and auto-registered.
use jolt_inlines_sha2 as _;

// Guest input pinned to the `e2e_profiling.rs` default (≈15M cycles).
//
// Setup (guest build, ELF decode + expansion, one un-timed trace for the row
// count) runs once outside the measurement; each iteration re-traces from the
// shared `JoltProgram`. Backend variants (`x86`, `x86_fast`) join as
// additional bench ids in this group.
fn bench(c: &mut Criterion) {
    let objective = TraceGenObjective::new(Sha2Chain::profiling_default());
    let setup = objective.setup();
    let mut group = c.benchmark_group(objective.name());
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(120));
    group.throughput(Throughput::Elements(setup.trace_len as u64));
    group.bench_function("reference", |b| b.iter(|| objective.run_reference(&setup)));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
