use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, Criterion};
use dory::arithmetic::Field;
use jolt_core::{
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn setup_inputs(n: u64) -> (MultilinearPolynomial<Fr>, Vec<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(n);
    let poly: MultilinearPolynomial<Fr> =
        MultilinearPolynomial::from((0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>());
    let eval_point = (0..(n as usize).log_2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    (poly, eval_point)
}

fn bench_all(c: &mut Criterion) {
    let num_vars = 1 << 12; // this is really num_coefs
    let (poly, eval_point) = setup_inputs(num_vars as u64);
    // Create a benchmark group
    let mut group = c.benchmark_group("evals");

    let id = format!("serial-{num_vars}");
    group.bench_function(id, |b| {
        b.iter(|| poly.evaluate(eval_point.as_slice()));
    });
    let id = format!("serial_fast-{num_vars}");
    group.bench_function(id, |b| {
        b.iter(|| poly.optimised_evaluate(eval_point.as_slice()));
    });

    group.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
