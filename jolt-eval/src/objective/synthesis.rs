/// Macro that generates a Criterion benchmark harness for a `PerfObjective`.
///
/// Uses `iter_batched` with `BatchSize::LargeInput` so that per-iteration
/// setup (e.g. polynomial clone) is excluded from the measurement.
///
/// # Usage
///
/// ```ignore
/// // Fast benchmark (default Criterion settings, type must impl Default):
/// jolt_eval::bench_objective!(BindLowToHighObjective);
///
/// // Slow benchmark with custom Criterion config:
/// jolt_eval::bench_objective!(
///     ProverTimeObjective::new(Fibonacci(100)),
///     config: sample_size(10), sampling_mode(Flat), measurement_time(30s)
/// );
/// ```
#[macro_export]
macro_rules! bench_objective {
    // Expression form with config methods
    ($obj_expr:expr, config: $($method:ident($($arg:expr),*)),* $(,)?) => {
        use $crate::PerfObjective as _;

        fn __bench(c: &mut ::criterion::Criterion) {
            let obj = $obj_expr;
            let mut group = c.benchmark_group(obj.name());
            $(
                group.$method($($arg),*);
            )*
            group.bench_function("prove", |b| {
                b.iter_batched(
                    || obj.setup(),
                    |setup| obj.run(setup),
                    ::criterion::BatchSize::LargeInput,
                );
            });
            group.finish();
        }

        ::criterion::criterion_group!(benches, __bench);
        ::criterion::criterion_main!(benches);
    };

    // Simple form: just a type (uses Default + default Criterion config)
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
