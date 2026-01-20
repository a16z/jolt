//! Benchmarks comparing algorithms for multilinear polynomial evaluation
//!
//! Source: https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
//!
//! This benchmark suite tests:
//! 1. Single polynomial evaluation with three algorithms across varying sparsity
//! 2. Batch evaluation of multiple polynomials at the same point
use ark_ff::Zero;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_core::{field::tracked_ark::TrackedFr as Fr, poly::dense_mlpoly::DensePolynomial};
use jolt_core::{field::JoltField, poly::multilinear_polynomial::PolynomialEvaluation};
use jolt_core::{poly::eq_poly::EqPolynomial, utils::math::Math};
use rayon::prelude::*;

/// Generate a sparse multilinear polynomial with controlled sparsity
///
/// # Arguments
/// * `n` - Number of coefficients (must be power of 2)
/// * `sparsity` - Fraction of coefficients that should be zero (0.0 to 1.0)
///
/// # Returns
/// Tuple of (sparse polynomial, random evaluation point with log2(n) variables)
///
/// # Example
/// `sparse_inputs(1024, 0.75)` creates a polynomial with ~75% zero coefficients
fn sparse_inputs(n: usize, sparsity: f64) -> (DensePolynomial<Fr>, Vec<Fr>) {
    assert!(n.is_power_of_two(), "n must be a power of 2");
    let mut rng = StdRng::seed_from_u64(123);

    let values: Vec<Fr> = (0..n)
        .map(|_| {
            if rng.gen::<f64>() < sparsity {
                Fr::zero()
            } else {
                <Fr as JoltField>::random(&mut rng)
            }
        })
        .collect();

    let poly = DensePolynomial::new(values);
    let eval_point = (0..n.log_2())
        .map(|_| <Fr as JoltField>::random(&mut rng))
        .collect();

    (poly, eval_point)
}

/// Setup inputs for batch polynomial evaluation benchmarks
///
/// # Arguments
/// * `n` - Number of coefficients per polynomial
/// * `batch_size` - Number of polynomials to evaluate
/// * `sparsity` - Fraction of zero coefficients in each polynomial
///
/// # Returns
/// Tuple of (vector of sparse polynomials, shared evaluation point)
fn setup_batch_inputs(
    n: usize,
    batch_size: usize,
    sparsity: f64,
) -> (Vec<DensePolynomial<Fr>>, Vec<Fr>) {
    let mut rng = StdRng::seed_from_u64(123);
    let eval_point: Vec<Fr> = (0..n.log_2())
        .map(|_| <Fr as JoltField>::random(&mut rng))
        .collect();

    let polys: Vec<DensePolynomial<Fr>> = (0..batch_size)
        .map(|_| sparse_inputs(n, sparsity).0)
        .collect();

    (polys, eval_point)
}

/// Benchmark three algorithms for single polynomial evaluation with varying sparsity
///
/// Compares:
/// 1. `dot_product` - Dot product the chi's with the evaluations over the hyper cube
/// 2. `evaluate` - First split the chi's and then do a a cache efficient dot product
/// 3. `inside_out` - For denese representations do not spend time doing dot product and then
///    computing chi's
///
/// Tests across different polynomial sizes (2^14, 2^16) and sparsity levels (20%, 50%, 75% zeros)
///  2^14 is configured to be serial and 2^16 is configured to parallel.
///
/// See: https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
fn benchmark_single_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_polynomial");

    for exp in [14, 16] {
        let num_vars = 1 << exp;
        for sparsity in [0.20, 0.50, 0.75] {
            group.bench_with_input(
                BenchmarkId::new("dot_product", format!("2^{exp}_s{sparsity}")),
                &(num_vars, sparsity),
                |b, params| {
                    let (n, s) = *params;
                    b.iter_with_setup(
                        || sparse_inputs(n, s),
                        |(poly, eval_point)| black_box(poly.evaluate_dot_product(&eval_point)),
                    )
                },
            );

            group.bench_with_input(
                BenchmarkId::new("evaluate", format!("2^{exp}_s{sparsity}")),
                &(num_vars, sparsity),
                |b, params| {
                    let (n, s) = *params;
                    b.iter_with_setup(
                        || sparse_inputs(n, s),
                        |(poly, eval_point)| black_box(poly.evaluate(&eval_point)),
                    )
                },
            );

            group.bench_with_input(
                BenchmarkId::new("inside_out", format!("2^{exp}_s{sparsity}")),
                &(num_vars, sparsity),
                |b, params| {
                    let (n, s) = *params;
                    b.iter_with_setup(
                        || sparse_inputs(n, s),
                        |(poly, eval_point)| black_box(poly.inside_out_evaluate(&eval_point)),
                    )
                },
            );
        }
    }
    group.finish();
}

// Benchmark batch evaluation of multiple polynomials at the same point
///
/// Compares:
/// 1. `dot_product` - Parallel evaluation using EqPolynomial precomputation
/// 2. `batch_evaluate` - A cache efficient dot product with SplitEqPolynomial instead of EqPolynomial
///
/// Tests with 50 polynomials of size 2^16 at varying sparsity levels.
/// Batch evaluation can amortize work across polynomials for better efficiency.
///
/// See: https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
fn benchmark_batch_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_polynomial");
    fn batch_dot_product(polys: &[DensePolynomial<Fr>], eval_point: &[Fr]) -> Vec<Fr> {
        let eq = EqPolynomial::evals(eval_point);
        polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi_low_optimized(&eq))
            .collect()
    }

    for exp in [16] {
        let num_vars = 1 << exp;
        for sparsity in [0.75, 0.50, 0.20] {
            {
                let batch_size = 50;
                group.bench_with_input(
                    BenchmarkId::new("dot_product", format!("2^{exp}_s{sparsity}_b{batch_size}")),
                    &(num_vars, sparsity, batch_size),
                    |b, params| {
                        let (n, s, bs) = *params;
                        b.iter_with_setup(
                            || setup_batch_inputs(n, bs, s),
                            |(polys, eval_point)| black_box(batch_dot_product(&polys, &eval_point)),
                        )
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(
                        "batch_evaluate",
                        format!("2^{exp}_s{sparsity}_b{batch_size}"),
                    ),
                    &(num_vars, sparsity, batch_size),
                    |b, params| {
                        let (n, s, bs) = *params;
                        b.iter_with_setup(
                            || setup_batch_inputs(n, bs, s),
                            |(polys, eval_point)| {
                                let poly_refs: Vec<&DensePolynomial<Fr>> = polys.iter().collect();
                                black_box(DensePolynomial::batch_evaluate::<Fr>(
                                    &poly_refs,
                                    &eval_point,
                                ))
                            },
                        )
                    },
                );
            }
        }
    }
    group.finish();
}
criterion_group!(
    benches,
    benchmark_single_evaluation,
    benchmark_batch_evaluation
);
criterion_main!(benches);
