#![allow(unused_results)]

//! Benchmark: direct Rayon+ToomCook fold vs ComputeBackend pairwise_reduce.
//!
//! Measures whether the `ComputeBackend` abstraction adds overhead compared
//! to the hand-written Rayon fold pattern used in the witness hot path
//! (`mles_product_sum.rs`).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compute::{BindingOrder, ComputeBackend, EqInput};
use jolt_cpu::{compile, toom_cook, CpuBackend};
use jolt_field::{Field, FieldAccumulator, Fr};
use jolt_compiler::{CompositionFormula, Factor, ProductTerm};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_field_vec(n: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

/// Helper: build a pure product-sum formula with `p` groups of `d` consecutive inputs.
fn product_sum_formula(d: usize, p: usize) -> CompositionFormula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    CompositionFormula::from_terms(terms)
}

/// Direct Rayon path: mimics the witness hot path from `mles_product_sum.rs`.
///
/// Uses `rayon::par_iter().fold().reduce()` with `FieldAccumulator` delayed
/// reduction, calling the hand-optimized `toom_cook::eval_prod_D_assign`.
fn direct_rayon_reduce_d4(inputs: &[Vec<Fr>], weights: &[Fr]) -> Vec<Fr> {
    use rayon::prelude::*;

    let half = inputs[0].len() / 2;
    let new_accs =
        || -> Vec<<Fr as Field>::Accumulator> { vec![<Fr as Field>::Accumulator::default(); 4] };

    let accs = (0..half)
        .into_par_iter()
        .fold(new_accs, |mut acc, i| {
            let pairs: [(Fr, Fr); 4] =
                core::array::from_fn(|k| (inputs[k][2 * i], inputs[k][2 * i + 1]));
            let mut endpoints = [Fr::zero(); 4];
            toom_cook::eval_prod_4_assign(&pairs, &mut endpoints);
            let w = weights[i];
            for (a, e) in acc.iter_mut().zip(endpoints.iter()) {
                a.fmadd(w, *e);
            }
            acc
        })
        .reduce(new_accs, |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(b) {
                ai.merge(bi);
            }
            a
        });

    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

fn direct_rayon_reduce_d8(inputs: &[Vec<Fr>], weights: &[Fr]) -> Vec<Fr> {
    use rayon::prelude::*;

    let half = inputs[0].len() / 2;
    let new_accs =
        || -> Vec<<Fr as Field>::Accumulator> { vec![<Fr as Field>::Accumulator::default(); 8] };

    let accs = (0..half)
        .into_par_iter()
        .fold(new_accs, |mut acc, i| {
            let pairs: [(Fr, Fr); 8] =
                core::array::from_fn(|k| (inputs[k][2 * i], inputs[k][2 * i + 1]));
            let mut endpoints = [Fr::zero(); 8];
            toom_cook::eval_prod_8_assign(&pairs, &mut endpoints);
            let w = weights[i];
            for (a, e) in acc.iter_mut().zip(endpoints.iter()) {
                a.fmadd(w, *e);
            }
            acc
        })
        .reduce(new_accs, |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(b) {
                ai.merge(bi);
            }
            a
        });

    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

fn direct_rayon_reduce_d16(inputs: &[Vec<Fr>], weights: &[Fr]) -> Vec<Fr> {
    use rayon::prelude::*;

    let half = inputs[0].len() / 2;
    let new_accs =
        || -> Vec<<Fr as Field>::Accumulator> { vec![<Fr as Field>::Accumulator::default(); 16] };

    let accs = (0..half)
        .into_par_iter()
        .fold(new_accs, |mut acc, i| {
            let pairs: [(Fr, Fr); 16] =
                core::array::from_fn(|k| (inputs[k][2 * i], inputs[k][2 * i + 1]));
            let mut endpoints = [Fr::zero(); 16];
            toom_cook::eval_prod_16_assign(&pairs, &mut endpoints);
            let w = weights[i];
            for (a, e) in acc.iter_mut().zip(endpoints.iter()) {
                a.fmadd(w, *e);
            }
            acc
        })
        .reduce(new_accs, |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(b) {
                ai.merge(bi);
            }
            a
        });

    accs.into_iter().map(FieldAccumulator::reduce).collect()
}

fn bench_rayon_vs_backend(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("rayon_vs_backend");

    for (d, direct_fn) in [
        (
            4usize,
            direct_rayon_reduce_d4 as fn(&[Vec<Fr>], &[Fr]) -> Vec<Fr>,
        ),
        (8, direct_rayon_reduce_d8),
        (16, direct_rayon_reduce_d16),
    ] {
        let formula = product_sum_formula(d, 1);
        let kernel = compile::<Fr>(&formula);

        for log_n in [16, 18, 20] {
            let n = 1usize << log_n;
            let half = n / 2;

            let inputs: Vec<Vec<Fr>> = (0..d)
                .map(|i| random_field_vec(n, 100 + i as u64 + d as u64 * 1000))
                .collect();
            let weights = random_field_vec(half, 200 + d as u64);
            let input_refs: Vec<&Vec<Fr>> = inputs.iter().collect();

            group.throughput(Throughput::Elements(half as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("direct_rayon/D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| {
                        black_box(direct_fn(&inputs, &weights));
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("backend/D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| {
                        black_box(backend.pairwise_reduce(
                            &input_refs,
                            EqInput::Weighted(&weights),
                            &kernel,
                            d,
                            BindingOrder::LowToHigh,
                        ));
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_rayon_vs_backend);
criterion_main!(benches);
