//! Example single-kernel instruction-count objective:
//! `callgrind:eq_evals:instructions`.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::EqPolynomial;

fn setup_point(vars: usize) -> Vec<Fr> {
    (0..vars)
        .map(|index| Fr::from_u64(2 * index as u64 + 1))
        .collect()
}

#[library_benchmark]
#[bench::vars_16(setup_point(16))]
fn bench_eq_evals(point: Vec<Fr>) -> Vec<Fr> {
    black_box(EqPolynomial::<Fr>::evals(&point, None))
}

library_benchmark_group!(
    name = eq_evals_group;
    benchmarks = bench_eq_evals
);

main!(library_benchmark_groups = eq_evals_group);
