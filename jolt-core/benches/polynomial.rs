use ark_bn254::Fr;
use ark_std::{rand::Rng, test_rng};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::sparse_interleaved_poly::{SparseCoefficient, SparseInterleavedPolynomial};
use jolt_core::subprotocols::sumcheck::Bindable;
use rayon::prelude::*;

fn random_dense_coeffs<F: JoltField>(rng: &mut impl Rng, num_vars: usize) -> Vec<F> {
    std::iter::repeat_with(|| F::random(rng))
        .take(1 << num_vars)
        .collect()
}

fn random_sparse_coeffs<F: JoltField>(
    rng: &mut impl Rng,
    batch_size: usize,
    num_vars: usize,
    density: f64,
) -> Vec<Vec<SparseCoefficient<F>>> {
    (0..batch_size)
        .map(|batch_index| {
            let mut coeffs: Vec<SparseCoefficient<F>> = vec![];
            for i in 0..(1 << num_vars) {
                if rng.gen_bool(density) {
                    coeffs.push((batch_index * (1 << num_vars) + i, F::random(rng)).into())
                }
            }
            coeffs
        })
        .collect()
}

fn benchmark_dense<F: JoltField>(c: &mut Criterion, num_vars: usize) {
    c.bench_function(
        &format!("DensePolynomial::bind {} variables", num_vars),
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
                    for i in 0..num_vars {
                        criterion::black_box(poly.bound_poly_var_top_par(&r[i]));
                    }
                },
            );
        },
    );
}

fn benchmark_dense_batch<F: JoltField>(c: &mut Criterion, num_vars: usize, batch_size: usize) {
    c.bench_function(
        &format!(
            "DensePolynomial::bind {} x {} variables",
            batch_size, num_vars
        ),
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
                    for i in 0..num_vars {
                        polys
                            .par_iter_mut()
                            .for_each(|poly| poly.bound_poly_var_bot(&r[i]))
                    }
                },
            );
        },
    );
}

fn benchmark_dense_interleaved<F: JoltField>(c: &mut Criterion, num_vars: usize) {
    c.bench_function(
        &format!("DenseInterleavedPolynomial::bind {} variables", num_vars),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_dense_coeffs(&mut rng, num_vars);
                    let poly = DenseInterleavedPolynomial::new(coeffs);
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (poly, r)
                },
                |(mut poly, r)| {
                    for i in 0..num_vars {
                        criterion::black_box(poly.bind(r[i]));
                    }
                },
            );
        },
    );
}

fn benchmark_sparse_interleaved<F: JoltField>(
    c: &mut Criterion,
    batch_size: usize,
    num_vars: usize,
    density: f64,
) {
    c.bench_function(
        &format!(
            "SparseInterleavedPolynomial::bind {} x {} variables, {}% ones",
            batch_size,
            num_vars,
            (1.0 - density) * 100.0
        ),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_sparse_coeffs(&mut rng, batch_size, num_vars, density);
                    let poly = SparseInterleavedPolynomial::new(coeffs, batch_size << num_vars);
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (poly, r)
                },
                |(mut poly, r)| {
                    for i in 0..num_vars {
                        criterion::black_box(poly.bind(r[i]));
                    }
                },
            );
        },
    );
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));

    benchmark_sparse_interleaved::<Fr>(&mut criterion, 64, 20, 0.1);
    benchmark_sparse_interleaved::<Fr>(&mut criterion, 128, 20, 0.1);
    benchmark_sparse_interleaved::<Fr>(&mut criterion, 64, 21, 0.1);
    benchmark_sparse_interleaved::<Fr>(&mut criterion, 128, 21, 0.1);

    // benchmark_dense::<Fr>(&mut criterion, 20);
    // benchmark_dense::<Fr>(&mut criterion, 22);
    // benchmark_dense::<Fr>(&mut criterion, 24);

    // benchmark_dense_interleaved::<Fr>(&mut criterion, 22);
    // benchmark_dense_interleaved::<Fr>(&mut criterion, 23);
    // benchmark_dense_interleaved::<Fr>(&mut criterion, 24);
    // benchmark_dense_interleaved::<Fr>(&mut criterion, 25);

    // benchmark_dense_batch::<Fr>(&mut criterion, 20, 4);
    // benchmark_dense_batch::<Fr>(&mut criterion, 20, 8);
    // benchmark_dense_batch::<Fr>(&mut criterion, 20, 16);
    // benchmark_dense_batch::<Fr>(&mut criterion, 20, 32);

    criterion.final_summary();
}
