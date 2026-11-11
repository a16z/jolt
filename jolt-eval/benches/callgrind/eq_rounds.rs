//! Deterministic instruction counts for little-endian eq expansion, Gruen
//! bind sweeps and degree-2, degree-3 and Toom rounds in both binding orders,
//! and tensor folding at 16 variables. `callgrind:eq_rounds:instructions`
//! sums all cases. Callgrind counts only the benchmark thread, so the tensor
//! fold runs in a one-thread Rayon pool that uses that thread as its worker.
//!
//! Opt-in: requires Valgrind and an `iai-callgrind-runner` matching the
//! workspace `iai-callgrind` version.

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main, LibraryBenchmarkConfig};
use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, PointRef, TensorEqTable};
use rayon::{ThreadPool, ThreadPoolBuilder};

fn setup_point(vars: usize) -> Vec<Fr> {
    (0..vars).map(|i| Fr::from_u64(2 * i as u64 + 1)).collect()
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

#[library_benchmark]
#[bench::vars_16_low_to_high(setup_gruen(BindingOrder::LowToHigh))]
#[bench::vars_16_high_to_low(setup_gruen(BindingOrder::HighToLow))]
fn bench_gruen_deg_2(input: (GruenSplitEqPolynomial<Fr>, Vec<Fr>)) -> GruenSplitEqPolynomial<Fr> {
    let (mut eq, challenges) = input;
    let q_zero = Fr::from_u64(2);
    let q_one = Fr::from_u64(5);
    for challenge in challenges {
        let (l_zero, l_one) = eq.current_linear_evals();
        let hint = l_zero * q_zero + l_one * q_one;
        black_box(eq.gruen_poly_deg_2(q_one, hint, || q_zero)).expect("consistent round hint");
        eq.bind(black_box(challenge));
    }
    black_box(eq)
}

#[library_benchmark]
#[bench::vars_16_low_to_high(setup_gruen(BindingOrder::LowToHigh))]
#[bench::vars_16_high_to_low(setup_gruen(BindingOrder::HighToLow))]
fn bench_gruen_deg_3(input: (GruenSplitEqPolynomial<Fr>, Vec<Fr>)) -> GruenSplitEqPolynomial<Fr> {
    let (mut eq, challenges) = input;
    let q_zero = Fr::from_u64(2);
    let q_one = Fr::from_u64(10);
    for challenge in challenges {
        let (l_zero, l_one) = eq.current_linear_evals();
        let hint = l_zero * q_zero + l_one * q_one;
        black_box(eq.gruen_poly_deg_3(q_zero, Fr::from_u64(5), hint, || q_one))
            .expect("consistent round hint");
        eq.bind(black_box(challenge));
    }
    black_box(eq)
}

#[library_benchmark]
#[bench::vars_16_low_to_high(setup_gruen(BindingOrder::LowToHigh))]
#[bench::vars_16_high_to_low(setup_gruen(BindingOrder::HighToLow))]
fn bench_gruen_toom(input: (GruenSplitEqPolynomial<Fr>, Vec<Fr>)) -> GruenSplitEqPolynomial<Fr> {
    let (mut eq, challenges) = input;
    let q_zero = Fr::from_u64(2);
    let q_one = Fr::from_u64(10);
    for challenge in challenges {
        let (l_zero, l_one) = eq.current_linear_evals();
        let hint = l_zero * q_zero + l_one * q_one;
        black_box(eq.gruen_poly_from_evals(&[q_one, Fr::from_u64(5)], hint, || q_zero))
            .expect("consistent round hint");
        eq.bind(black_box(challenge));
    }
    black_box(eq)
}

fn setup_tensor() -> (TensorEqTable<Fr>, Vec<Fr>, ThreadPool) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .expect("one-thread pool on the benchmark thread");
    (
        TensorEqTable::new(&setup_point(16)),
        setup_point(1 << 16),
        pool,
    )
}

#[library_benchmark]
#[bench::vars_16(setup_tensor())]
fn bench_tensor_fold(input: (TensorEqTable<Fr>, Vec<Fr>, ThreadPool)) -> Fr {
    let (eq, values, pool) = input;
    black_box(pool.install(|| {
        eq.par_fold_out_in(
            Fr::zero,
            |acc, row, _, weight| *acc += weight * values[row],
            |_, weight, inner| weight * inner,
            |left, right| left + right,
        )
    }))
}

library_benchmark_group!(
    name = eq_rounds_group;
    benchmarks = bench_eq_evals_le, bench_gruen_bind, bench_gruen_deg_2, bench_gruen_deg_3, bench_gruen_toom, bench_tensor_fold
);

main!(
    config = LibraryBenchmarkConfig::default().env("RAYON_NUM_THREADS", "1");
    library_benchmark_groups = eq_rounds_group
);
