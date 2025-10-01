#![allow(clippy::uninlined_format_args)]
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
    let mut group = c.benchmark_group("evals");
    //group.measurement_time(std::time::Duration::from_secs(60));

    for &exp in &[12, 14, 16, 18, 20, 22, 24] {
        let num_vars = 1 << exp; // 2^exp
        let (poly, eval_point) = setup_inputs(num_vars as u64);

        let id_dot = format!("split-eval-{}", exp);
        group.bench_function(&id_dot, |b| b.iter(|| poly.evaluate(eval_point.as_slice())));

        let id_dot = format!("dot-product-{}", exp);
        group.bench_function(&id_dot, |b| {
            b.iter(|| poly.evaluate_dot_product(eval_point.as_slice()))
        });

        //let id_opt = format!("inside-out-{}", exp);
        //group.bench_function(&id_opt, |b| b.iter(|| poly(eval_point.as_slice())));
    }

    group.finish();
}
criterion_group!(benches, bench_all);
criterion_main!(benches);
