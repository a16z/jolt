//! Deterministic instruction-count microbenchmark over the eq-table
//! expansion (`EqPolynomial::evals`) — one of the sumcheck inner loop's
//! dominant leaf costs. Measured as `callgrind:eq_evals:instructions`.
//!
//! Opt-in: needs Valgrind and `cargo install iai-callgrind-runner` (version
//! matching the workspace `iai-callgrind`). 16 variables keeps the expansion
//! on the serial path, so the instruction count is exactly reproducible.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, PointRef, TensorEqTable};

fn setup_point(vars: usize) -> Vec<Fr> {
    (0..vars).map(|i| Fr::from_u64(2 * i as u64 + 1)).collect()
}

#[library_benchmark]
#[bench::vars_16(setup_point(16))]
fn bench_eq_evals(point: Vec<Fr>) -> Vec<Fr> {
    black_box(EqPolynomial::<Fr>::evals(&point, None))
}

#[library_benchmark]
#[bench::vars_16_le(setup_point(16))]
fn bench_eq_evals_le(point: Vec<Fr>) -> Vec<Fr> {
    black_box(EqPolynomial::<Fr>::evals(
        &PointRef::low_to_high(&point),
        None,
    ))
}

fn setup_gruen(order: BindingOrder) -> (GruenSplitEqPolynomial<Fr>, Vec<Fr>) {
    (
        GruenSplitEqPolynomial::new(&setup_point(16), order),
        (0..16).map(|i| Fr::from_u64(2 * i + 37)).collect(),
    )
}

#[library_benchmark]
#[bench::vars_16_low_to_high(setup_gruen(BindingOrder::LowToHigh))]
#[bench::vars_16_high_to_low(setup_gruen(BindingOrder::HighToLow))]
fn bench_gruen_bind(input: (GruenSplitEqPolynomial<Fr>, Vec<Fr>)) -> GruenSplitEqPolynomial<Fr> {
    let (mut eq, challenges) = input;
    for challenge in challenges {
        eq.bind(black_box(challenge));
    }
    black_box(eq)
}

fn setup_tensor() -> (TensorEqTable<Fr>, Vec<Fr>) {
    (TensorEqTable::new(&setup_point(16)), setup_point(1 << 16))
}

#[library_benchmark]
#[bench::vars_16(setup_tensor())]
fn bench_tensor_fold(input: (TensorEqTable<Fr>, Vec<Fr>)) -> Fr {
    let (eq, values) = input;
    black_box(eq.par_fold_out_in(
        Fr::zero,
        |acc, row, _, weight| *acc += weight * values[row],
        |_, weight, inner| weight * inner,
        |left, right| left + right,
    ))
}

library_benchmark_group!(
    name = eq_evals_group;
    benchmarks = bench_eq_evals, bench_eq_evals_le, bench_gruen_bind, bench_tensor_fold
);

main!(
    config = LibraryBenchmarkConfig::default().env("RAYON_NUM_THREADS", "1");
    library_benchmark_groups = eq_evals_group
);
