/*
use core::num;

use ark_ec::AffineRepr;
use iai_callgrind::{main, library_benchmark, library_benchmark_group};
use jolt_core::{msm::, poly::dense_mlpoly::};

/*
     value = length of MSM
 */
fn msm_setup<C: AffineRepr>(num_points: u64) -> (Vec<C>, Vec<C::ScalarField>) {
    // Define the elliptic curve you want to work with.
    // For example, let's use the BN254 curve.

    // Initialize a random number generator.
    let mut rng = rand::thread_rng();

    // Generate a vector of random affine points on the curve.
    (vec![C::ScalarField::rand(&mut rng); num_points], vec![C::rand(&mut rng); num_points])
}

// length of poly
fn poly_setup(value: u64) -> () {

}

#[library_benchmark]
#[bench::long(msm_setup(30))]
fn bench_msm(input: (Vec<F>, Vec<AffineRepr>)) -> u64 {
    black_box(VariableBaseMSM::msm(&G::normalize_batch(&input.0), &input.1).unwrap());
}

// Poly benches fix vars of polys of size 12-32
#[library_benchmark]
#[bench::long(poly_setup(30))]
fn bench_polynomial_binding(j: F, poly: DensePolynomial<F>) -> u64 {
    black_box(poly.bound_poly_var_top(&j))
}

#[library_benchmark]
#[bench::long(poly_setup(30))]
fn bench_polynomial_evaluation(j: F, value: DensePolynomial<F>) -> u64 {
    black_box(poly.bound_poly_var_top(&j))
}

main!(
    callgrind_args = "toggle-collect=util::*";
    functions = bench_msm, bench_polynomial_binding, bench_polynomial_evaluation
);

fn some_setup_func(value: u64) -> u64 {
    value
}

#[library_benchmark]
#[bench::long(some_setup_func(30))]
fn bench_fibonacci(value: u64) -> u64 {
    black_box(fibonacci(value))
}
*/

use iai_callgrind::{main, library_benchmark_group, library_benchmark};
use std::hint::black_box;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[library_benchmark]
#[bench::short(10)]
#[bench::long(30)]
fn bench_fibonacci(value: u64) -> u64 {
    black_box(fibonacci(value))
}

library_benchmark_group!(
    name = bench_fibonacci_group;
    benchmarks = bench_fibonacci
);

main!(library_benchmark_groups = bench_fibonacci_group);