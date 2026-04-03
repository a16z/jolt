/// Macro that generates a Criterion benchmark harness for a `PerfObjective`.
///
/// Takes a concrete `PerfObjective` expression. Setup is performed once;
/// Criterion calls `run()` repeatedly with statistical rigor.
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
            let mut setup = obj.setup();
            c.bench_function(obj.name(), |b| {
                b.iter(|| obj.run(&mut setup));
            });
        }

        ::criterion::criterion_group!(benches, __bench);
        ::criterion::criterion_main!(benches);
    };
}
