//! Benchmarks comparing Metal kernel throughput vs CPU.
//!
//! Tuned for fast iteration: sample_size=10, two sizes only (2^14 = near-crossover,
//! 2^20 = saturated). Focus on the hot path (pairwise_reduce, sumcheck_round).
//!
//! Run all:     cargo bench -p jolt-metal --bench metal_vs_cpu -q
//! Run subset:  cargo bench -p jolt-metal --bench metal_vs_cpu -q -- pairwise_reduce

#![cfg(target_os = "macos")]
#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compiler::{Factor, Formula, ProductTerm};
use jolt_compute::{BindingOrder, ComputeBackend, EqInput};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_metal::MetalBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn random_fr(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

/// Two sizes: near-crossover and saturated.
const SMALL: usize = 1 << 14;
const LARGE: usize = 1 << 20;

fn fast_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
}

fn bench_pairwise_reduce(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce");

    for &d in &[4usize, 8] {
        let formula = product_sum_formula(d, 1);

        let cpu_k = jolt_cpu::compile::<Fr>(&formula);
        let mtl_k = metal.compile_kernel::<Fr>(&formula);

        for &n in &[SMALL, LARGE] {
            let mut rng = StdRng::seed_from_u64(100 + n as u64);
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();
            let weights = random_fr(&mut rng, n / 2);

            let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
            let mtl_w = metal.upload(&weights);
            let cpu_w = cpu.upload(&weights);

            let label = format!("D{d}/2^{}", n.trailing_zeros());
            group.throughput(Throughput::Elements((n / 2) as u64));

            let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        metal.pairwise_reduce(
                            &mtl_refs,
                            EqInput::Weighted(&mtl_w),
                            &mtl_k,
                            &[],
                            formula.degree(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );

            let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        cpu.pairwise_reduce(
                            &cpu_refs,
                            EqInput::Weighted(&cpu_w),
                            &cpu_k,
                            &[],
                            formula.degree(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_pairwise_reduce_unweighted(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce_unw");

    for &d in &[4usize, 8] {
        let formula = product_sum_formula(d, 1);

        let cpu_k = jolt_cpu::compile::<Fr>(&formula);
        let mtl_k = metal.compile_kernel::<Fr>(&formula);

        for &n in &[SMALL, LARGE] {
            let mut rng = StdRng::seed_from_u64(600 + n as u64);
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();

            let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
            let label = format!("D{d}/2^{}", n.trailing_zeros());
            group.throughput(Throughput::Elements((n / 2) as u64));

            let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        metal.pairwise_reduce(
                            &mtl_refs,
                            EqInput::Unit,
                            &mtl_k,
                            &[],
                            formula.degree(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );

            let cpu_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        cpu.pairwise_reduce(
                            &cpu_refs,
                            EqInput::Unit,
                            &cpu_k,
                            &[],
                            formula.degree(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_sumcheck_round(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("sumcheck_round");

    for &d in &[4usize, 8] {
        let n_inputs = d;
        let formula = product_sum_formula(d, 1);

        for &log_n in &[14u32, 20] {
            let n = 1usize << log_n;
            let rounds = 4;
            let mut rng = StdRng::seed_from_u64(800 + log_n as u64 + d as u64);
            let data: Vec<Vec<Fr>> = (0..n_inputs).map(|_| random_fr(&mut rng, n)).collect();
            let scalar = Fr::random(&mut rng);

            let label = format!("D{d}/2^{log_n}/{rounds}r");
            group.throughput(Throughput::Elements(n as u64));

            let mtl_k = metal.compile_kernel::<Fr>(&formula);
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<_> = data.iter().map(|v| metal.upload(v)).collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = metal.pairwise_reduce(
                                &refs,
                                EqInput::Unit,
                                &mtl_k,
                                &[],
                                formula.degree(),
                                BindingOrder::LowToHigh,
                            );
                            bufs = metal.interpolate_pairs_batch(bufs, scalar);
                        }
                        bufs
                    });
                },
            );

            let cpu_k = jolt_cpu::compile::<Fr>(&formula);
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<_> = data.iter().map(|v| cpu.upload(v)).collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = cpu.pairwise_reduce(
                                &refs,
                                EqInput::Unit,
                                &cpu_k,
                                &[],
                                formula.degree(),
                                BindingOrder::LowToHigh,
                            );
                            bufs = cpu.interpolate_pairs_batch(bufs, scalar);
                        }
                        bufs
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_interpolate(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("interpolate");

    for &n in &[SMALL, LARGE] {
        let mut rng = StdRng::seed_from_u64(500 + n as u64);
        let data = random_fr(&mut rng, n);
        let scalar = Fr::random(&mut rng);
        let label = format!("2^{}", n.trailing_zeros());

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("metal/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let buf = metal.upload(&data);
                    metal.interpolate_pairs(buf, scalar)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("cpu/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let buf = cpu.upload(&data);
                    cpu.interpolate_pairs(buf, scalar)
                });
            },
        );
    }
    group.finish();
}

/// Sumcheck round using H2L in-place binding (no buffer allocation per round).
fn bench_sumcheck_round_h2l(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("sumcheck_round_h2l");

    for &d in &[4usize, 8] {
        let n_inputs = d;
        let formula = product_sum_formula(d, 1);

        for &log_n in &[14u32, 20] {
            let n = 1usize << log_n;
            let rounds = 4;
            let mut rng = StdRng::seed_from_u64(800 + log_n as u64 + d as u64);
            let data: Vec<Vec<Fr>> = (0..n_inputs).map(|_| random_fr(&mut rng, n)).collect();
            let scalar = Fr::random(&mut rng);

            let label = format!("D{d}/2^{log_n}/{rounds}r");
            group.throughput(Throughput::Elements(n as u64));

            let mtl_k = metal.compile_kernel::<Fr>(&formula);
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<_> = data.iter().map(|v| metal.upload(v)).collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = metal.pairwise_reduce(
                                &refs,
                                EqInput::Unit,
                                &mtl_k,
                                &[],
                                formula.degree(),
                                BindingOrder::HighToLow,
                            );
                            metal.interpolate_pairs_batch_inplace(
                                &mut bufs,
                                scalar,
                                BindingOrder::HighToLow,
                            );
                        }
                        bufs
                    });
                },
            );

            let cpu_k = jolt_cpu::compile::<Fr>(&formula);
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<_> = data.iter().map(|v| cpu.upload(v)).collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = cpu.pairwise_reduce(
                                &refs,
                                EqInput::Unit,
                                &cpu_k,
                                &[],
                                formula.degree(),
                                BindingOrder::HighToLow,
                            );
                            cpu.interpolate_pairs_batch_inplace(
                                &mut bufs,
                                scalar,
                                BindingOrder::HighToLow,
                            );
                        }
                        bufs
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = fast_config();
    targets =
        bench_pairwise_reduce,
        bench_pairwise_reduce_unweighted,
        bench_sumcheck_round,
        bench_sumcheck_round_h2l,
        bench_interpolate,
}
criterion_main!(benches);
