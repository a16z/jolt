use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_std::{UniformRand, test_rng};
use ark_bn254::{G1Projective, Fr};
use iai_callgrind::{main, library_benchmark, library_benchmark_group};
use std::hint::black_box;
use jolt_core::{poly::dense_mlpoly::DensePolynomial, msm::VariableBaseMSM};

fn msm_setup<G: CurveGroup>(num_points: usize) -> (Vec<G>, Vec<G::ScalarField>) {
    let mut rng = test_rng();

    // Generate a vector of random affine points on the curve.
    (vec![G::rand(&mut rng); num_points], vec![G::ScalarField::rand(&mut rng); num_points])
}

fn bound_poly_setup<F: PrimeField>(size: usize) -> (DensePolynomial<F>, F) {
    let mut rng = test_rng();

    (DensePolynomial::new(vec![F::rand(&mut rng); size]), F::rand(&mut rng))
} 

fn eval_poly_setup<F: PrimeField>(size: usize) -> (DensePolynomial<F>, Vec<F>) {
    let mut rng = test_rng();

    let poly = DensePolynomial::new(vec![F::rand(&mut rng); size]);
    let points = vec![F::rand(&mut rng); poly.get_num_vars()];
    (poly, points)
} 

#[library_benchmark]
#[bench::long(msm_setup::<G1Projective>(4096))]
fn bench_msm<G: CurveGroup>(input: (Vec<G>, Vec<G::ScalarField>)) -> G {
    black_box(VariableBaseMSM::msm(&G::normalize_batch(&input.0), &input.1).unwrap())
}

#[library_benchmark]
#[bench::long(bound_poly_setup::<Fr>(4096))]
fn bench_polynomial_binding<F: PrimeField>(input: (DensePolynomial<F>, F)) {
    let (mut poly, val) = input;
    black_box(poly.bound_poly_var_top(&val));
}

#[library_benchmark]
#[bench::long(eval_poly_setup::<Fr>(4096))]
fn bench_polynomial_evaluate<F: PrimeField>(input: (DensePolynomial<F>, Vec<F>)) -> F {
    let (poly, points) = input;
    black_box(poly.evaluate(&points))
}

library_benchmark_group!(
    name = jolt_core_ops;
    benchmarks = bench_msm, bench_polynomial_binding, bench_polynomial_evaluate
);

main!(library_benchmark_groups = jolt_core_ops);