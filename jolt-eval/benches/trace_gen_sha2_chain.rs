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
// additional bench ids in this group (present below on x86_64 Linux).
fn bench(c: &mut Criterion) {
    // The tracer env-dispatches to parallel mode; pin serial so
    // measurements are environment-independent.
    std::env::remove_var("TRACER_PARALLEL");
    let objective = TraceGenObjective::new(Sha2Chain::profiling_default());
    let setup = objective.setup();
    let mut group = c.benchmark_group(objective.name());
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(std::time::Duration::from_secs(120));
    group.throughput(Throughput::Elements(setup.trace_len as u64));
    // Assert the row count rather than discard it: `Throughput::Elements` is
    // fixed at `setup.trace_len`, so any id in this group that produced a
    // different number of rows would report MHz against the wrong
    // denominator. Row equality is AC5's job; this only keeps the units
    // honest — and it applies most to the backend ids below.
    group.bench_function("reference", |b| {
        b.iter(|| assert_eq!(objective.run_reference(&setup), setup.trace_len))
    });
    // Same group and throughput so the pair decomposes the seam: `reference`
    // is raw tracing plus the Cycle to TraceRow conversion, `reference_raw` is
    // raw tracing alone. The AOT backend emits TraceRow directly and so skips
    // the difference; AC8/AC9 ratios need both to be attributable.
    group.bench_function("reference_raw", |b| {
        b.iter(|| assert_eq!(objective.run_reference_raw(&setup), setup.trace_len))
    });
    // The AOT backend's arms exist only where it has native codegen;
    // elsewhere NativeBackend is the interpreter and these would duplicate
    // the reference id. One backend outside the measured region, warmed with
    // an un-timed run: iterations must reuse the compile cache, not pay the
    // one-time AOT compile per iteration.
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
