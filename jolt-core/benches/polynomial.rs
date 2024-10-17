use ark_bn254::Fr;
use ark_std::{rand::Rng, test_rng};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::sparse_interleaved_poly::{SparseCoefficient, SparseInterleavedPolynomial};

fn random_dense_coeffs<F: JoltField>(rng: &mut impl Rng, num_vars: usize) -> Vec<F> {
    std::iter::repeat_with(|| F::random(rng))
        .take(1 << num_vars)
        .collect()
}

fn random_sparse_coeffs<F: JoltField>(
    rng: &mut impl Rng,
    num_vars: usize,
    density: f64,
) -> Vec<SparseCoefficient<F>> {
    let mut coeffs = vec![];
    for i in 0..(2 << num_vars) {
        if rng.gen_bool(density) {
            coeffs.push((i, F::random(rng)).into())
        }
    }
    coeffs
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

fn benchmark_sparse_interleaved<F: JoltField>(c: &mut Criterion, num_vars: usize, density: f64) {
    c.bench_function(
        &format!(
            "SparseInterleavedPolynomial::bind {} variables, {}% ones",
            num_vars,
            (1.0 - density) * 100.0
        ),
        |b| {
            b.iter_with_setup(
                || {
                    let mut rng = test_rng();
                    let coeffs = random_sparse_coeffs(&mut rng, num_vars, density);
                    let poly = SparseInterleavedPolynomial::new(coeffs, 2 << num_vars);
                    let r: Vec<F> = std::iter::repeat_with(|| F::random(&mut rng))
                        .take(num_vars)
                        .collect();
                    (poly, r)
                },
                |(mut poly, r)| {
                    for i in 0..num_vars {
                        criterion::black_box(poly.bind_slices(r[i]));
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

    benchmark_sparse_interleaved::<Fr>(&mut criterion, 24, 0.1);
    benchmark_sparse_interleaved::<Fr>(&mut criterion, 25, 0.1);
    // benchmark_sparse_interleaved::<Fr>(&mut criterion, 26, 0.1);
    // benchmark_sparse_interleaved::<Fr>(&mut criterion, 27, 0.1);

    // benchmark_dense::<Fr>(&mut criterion, 25);
    // benchmark_dense::<Fr>(&mut criterion, 26);
    // benchmark_dense::<Fr>(&mut criterion, 27);
    // benchmark_dense::<Fr>(&mut criterion, 28);

    criterion.final_summary();
}
