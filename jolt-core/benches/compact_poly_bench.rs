#![allow(clippy::uninlined_format_args)]
use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, Criterion};
use dory::arithmetic::Field;
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn setup_u8_inputs(n: usize) -> (MultilinearPolynomial<Fr>, Vec<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(n as u64);
    let coeffs: Vec<u8> = (0..n).map(|_| rand::random::<u8>()).collect();

    let poly = MultilinearPolynomial::U8Scalars(
        jolt_core::poly::compact_polynomial::CompactPolynomial::from_coeffs(coeffs),
    );

    let eval_point = (0..(n as f64).log2() as usize)
        .map(|_| Fr::random(&mut rng))
        .collect();

    (poly, eval_point)
}
fn bench_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("evals_u8");
    group.measurement_time(std::time::Duration::from_secs(60));

    for &exp in &[16, 18, 20] {
        let num_vars = 1 << exp; // 2^exp
        let (poly, eval_point) = setup_u8_inputs(num_vars);

        let id_dot = format!("u8-dot-product-{}", exp);
        group.bench_function(&id_dot, |b| {
            b.iter(|| poly.evaluate_dot_product(&eval_point))
        });

        let id_opt = format!("u8-inside-out-{}", exp);
        group.bench_function(&id_opt, |b| b.iter(|| poly.evaluate(&eval_point)));
    }

    group.finish();
}

criterion_group!(benches, bench_u8);
criterion_main!(benches);
