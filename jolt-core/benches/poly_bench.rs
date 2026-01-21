#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// Setup benchmark inputs for multilinear polynomial evaluation
///
/// Creates a random dense multilinear polynomial and evaluation point
///
/// # Arguments
/// * `n` - Number of coefficients in the polynomial (determines size)
///
/// # Returns
/// Tuple of (polynomial, evaluation_point) where evaluation_point has log2(n) variables
fn setup_inputs(n: u64) -> (MultilinearPolynomial<Fr>, Vec<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(n);
    let poly: MultilinearPolynomial<Fr> = MultilinearPolynomial::from(
        (0..n)
            .map(|_| <Fr as JoltField>::random(&mut rng))
            .collect::<Vec<_>>(),
    );
    let eval_point = (0..(n as usize).log_2())
        .map(|_| <Fr as JoltField>::random(&mut rng))
        .collect::<Vec<_>>();

    (poly, eval_point)
}

/// Benchmark comparing two algorithms for multilinear polynomial evaluation
///
/// Compares performance of:
/// 1. `evaluate()` - Standard split-and-merge algorithm
/// 2. `evaluate_dot_product()` - Dot product based algorithm
///
/// Both algorithms evaluate dense multilinear polynomials (no sparse coefficients)
/// across different polynomial sizes (2^12 to 2^24 coefficients)
fn bench_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("evals");

    for &exp in &[12, 14, 16, 18, 20, 22, 24] {
        let num_vars = 1 << exp; // 2^exp
        let (poly, eval_point) = setup_inputs(num_vars as u64);

        let id_dot = format!("split-eval-{}", exp);
        group.bench_function(&id_dot, |b| b.iter(|| poly.evaluate(eval_point.as_slice())));

        let id_dot = format!("dot-product-{}", exp);
        group.bench_function(&id_dot, |b| {
            b.iter(|| poly.evaluate_dot_product(eval_point.as_slice()))
        });
    }

    group.finish();
}
criterion_group!(benches, bench_all);
criterion_main!(benches);
