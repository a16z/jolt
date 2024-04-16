use iai_callgrind::{main, library_benchmark, library_benchmark_group};

fn msm_setup(value: u64) -> () {
    value
}

fn poly_setup(value: u64) -> () {
    value
}

#[library_benchmark]
#[bench::long(msm_setup(30))]
fn bench_msm(scalars: Vec<F>, points: Vec<Affine<F>>) -> u64 {
    black_box(msm(value))
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