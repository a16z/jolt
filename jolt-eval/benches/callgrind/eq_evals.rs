//! Deterministic instruction-count microbenchmark over the eq-table
//! expansion (`EqPolynomial::evals`) — one of the sumcheck inner loop's
//! dominant leaf costs. Measured as `callgrind:eq_evals:instructions`.
//!
//! Opt-in: needs Valgrind and `cargo install iai-callgrind-runner` (version
//! matching the workspace `iai-callgrind`). 16 variables keeps the expansion
//! on the serial path, so the instruction count is exactly reproducible.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use jolt_field::{Fr, Ring};
use jolt_poly::EqPolynomial;

fn setup_point(vars: usize) -> Vec<Fr> {
    (0..vars).map(|i| Fr::from_u64(2 * i as u64 + 1)).collect()
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
