use ark_bn254::Fr;
use ark_std::{rand::Rng, test_rng};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::compact_polynomial::CompactPolynomial;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use rayon::prelude::*;

fn random_dense_coeffs<F: JoltField>(rng: &mut impl Rng, num_vars: usize) -> Vec<F> {
    std::iter::repeat_with(|| F::random(rng))
        .take(1 << num_vars)
        .collect()
}

fn random_compact_coeffs(rng: &mut impl Rng, num_vars: usize) -> Vec<u8> {
    std::iter::repeat_with(|| rng.gen())
        .take(1 << num_vars)
        .collect()
}

fn benchmark_dense<F: JoltField>(c: &mut Criterion, num_vars: usize) {
    c.bench_function(
        &format!("DensePolynomial::bind {num_vars} variables"),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_dense_coeffs(&mut rng, num_vars);
                    let poly = DensePolynomial::new(coeffs);
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (poly, r)
                },
                |(mut poly, r)| {
                    (0..num_vars).for_each(|i| {
                        poly.bound_poly_var_top(&r[i]);
                        criterion::black_box(());
                    });
                },
            );
        },
    );
}

fn benchmark_dense_batch<F: JoltField>(c: &mut Criterion, num_vars: usize, batch_size: usize) {
    c.bench_function(
        &format!("DensePolynomial::bind {batch_size} x {num_vars} variables"),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let mut polys = vec![];
                    for _ in 0..batch_size {
                        let coeffs = random_dense_coeffs(&mut rng, num_vars);
                        polys.push(DensePolynomial::new(coeffs));
                    }
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (polys, r)
                },
                |(mut polys, r)| {
                    (0..num_vars).for_each(|i| {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bound_poly_var_bot(&r[i]))
                    });
                },
            );
        },
    );
}

fn benchmark_compact<F: JoltField>(
    c: &mut Criterion,
    num_vars: usize,
    binding_order: BindingOrder,
) {
    c.bench_function(
        &format!("CompactPolynomial::bind {num_vars} variables {binding_order:?} binding order"),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_compact_coeffs(&mut rng, num_vars);
                    let poly = CompactPolynomial::from_coeffs(coeffs);
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (poly, r)
                },
                |(mut poly, r)| {
                    r.into_iter().for_each(|r_i| {
                        poly.bind_parallel(r_i, binding_order);
                        criterion::black_box(());
                    });
                },
            );
        },
    );
}

fn benchmark_dense_parallel<F: JoltField>(
    c: &mut Criterion,
    num_vars: usize,
    binding_order: BindingOrder,
) {
    c.bench_function(
        &format!("DensePolynomial::bind_parallel {num_vars} variables {binding_order:?}"),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_dense_coeffs(&mut rng, num_vars);
                    let poly = DensePolynomial::new(coeffs);
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (poly, r)
                },
                |(mut poly, r)| {
                    (0..num_vars).for_each(|i| {
                        poly.bind_parallel(r[i], binding_order);
                        criterion::black_box(());
                    });
                },
            );
        },
    );
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));

    benchmark_dense::<Fr>(&mut criterion, 20);
    benchmark_dense::<Fr>(&mut criterion, 22);
    benchmark_dense::<Fr>(&mut criterion, 24);

    benchmark_dense_batch::<Fr>(&mut criterion, 20, 4);
    benchmark_dense_batch::<Fr>(&mut criterion, 20, 8);
    benchmark_dense_batch::<Fr>(&mut criterion, 20, 16);
    benchmark_dense_batch::<Fr>(&mut criterion, 20, 32);

    benchmark_dense_parallel::<Fr>(&mut criterion, 22, BindingOrder::LowToHigh);
    benchmark_dense_parallel::<Fr>(&mut criterion, 24, BindingOrder::LowToHigh);
    benchmark_dense_parallel::<Fr>(&mut criterion, 26, BindingOrder::LowToHigh);

    benchmark_dense_parallel::<Fr>(&mut criterion, 22, BindingOrder::HighToLow);
    benchmark_dense_parallel::<Fr>(&mut criterion, 24, BindingOrder::HighToLow);
    benchmark_dense_parallel::<Fr>(&mut criterion, 26, BindingOrder::HighToLow);

    benchmark_compact::<Fr>(&mut criterion, 22, BindingOrder::LowToHigh);
    benchmark_compact::<Fr>(&mut criterion, 24, BindingOrder::LowToHigh);
    benchmark_compact::<Fr>(&mut criterion, 26, BindingOrder::LowToHigh);

    criterion.final_summary();
}
