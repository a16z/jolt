/// Macro that generates a Criterion benchmark harness for a `PerfObjective`.
///
/// Uses `iter_batched` with `BatchSize::LargeInput` so that per-iteration
/// setup (e.g. polynomial clone) is excluded from the measurement.
///
/// # Usage
///
/// ```ignore
/// // benches/bind_parallel_low_to_high.rs
/// use jolt_eval::objective::bind_bench::BindLowToHighObjective;
/// jolt_eval::bench_objective!(BindLowToHighObjective);
/// ```
#[macro_export]
macro_rules! bench_objective {
    ($obj_ty:ty) => {
        use $crate::PerfObjective as _;

        fn __bench(c: &mut ::criterion::Criterion) {
            let obj = <$obj_ty>::default();
            c.bench_function(obj.name(), |b| {
                b.iter_batched(
                    || obj.setup(),
                    |setup| obj.run(setup),
                    ::criterion::BatchSize::LargeInput,
                );
            });
        }

        ::criterion::criterion_group!(benches, __bench);
        ::criterion::criterion_main!(benches);
    };
}
