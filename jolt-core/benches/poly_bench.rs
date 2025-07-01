use ark_bn254::Fr;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::poly::helpers::{evals_parallel, evals_parallel_dynamic, evals_serial};

fn setup_inputs(n: usize) -> Vec<Fr> {
    let mut rng = test_rng();
    (0..n).map(|_| Fr::rand(&mut rng)).collect()
}

fn bench_all(c: &mut Criterion) {
    let z = setup_inputs(18);
    // Create a benchmark group
    let mut group = c.benchmark_group("evals");

    group.bench_function("serial", |b| {
        b.iter(|| evals_serial(&z, None));
    });
    group.bench_function("parallel", |b| {
        b.iter(|| evals_parallel(&z, None));
    });
    group.bench_function("dynamic", |b| {
        b.iter(|| evals_parallel_dynamic(&z, None));
    });

    group.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
