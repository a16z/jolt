//! Benchmarks comparing Metal kernel throughput vs CPU.
//!
//! Tuned for fast iteration: sample_size=10, two sizes only (2^14 = near-crossover,
//! 2^20 = saturated). Focus on the hot path (reduce, sumcheck_round).
//!
//! Run all:     cargo bench -p jolt-metal --bench metal_vs_cpu -q
//! Run subset:  cargo bench -p jolt-metal --bench metal_vs_cpu -q -- reduce

#![cfg(target_os = "macos")]
#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{BindingOrder, Factor, Formula, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
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

fn dense_spec(formula: &Formula, order: BindingOrder) -> KernelSpec {
    KernelSpec::new(formula.clone(), Iteration::Dense, order)
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

fn bench_reduce(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("reduce");

    for &d in &[4usize, 8] {
        let formula = product_sum_formula(d, 1);
        let spec = dense_spec(&formula, BindingOrder::LowToHigh);

        let cpu_k = cpu.compile::<Fr>(&spec);
        let mtl_k = metal.compile::<Fr>(&spec);

        for &n in &[SMALL, LARGE] {
            let mut rng = StdRng::seed_from_u64(100 + n as u64);
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();

            let mtl_dev: Vec<Buf<MetalBackend, Fr>> = inputs
                .iter()
                .map(|v| DeviceBuffer::Field(metal.upload(v)))
                .collect();
            let mtl_refs: Vec<&Buf<MetalBackend, Fr>> = mtl_dev.iter().collect();

            let cpu_dev: Vec<Buf<CpuBackend, Fr>> = inputs
                .iter()
                .map(|v| DeviceBuffer::Field(cpu.upload(v)))
                .collect();
            let cpu_refs: Vec<&Buf<CpuBackend, Fr>> = cpu_dev.iter().collect();

            let label = format!("D{d}/2^{}", n.trailing_zeros());
            group.throughput(Throughput::Elements((n / 2) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| metal.reduce(&mtl_k, &mtl_refs, &[]));
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| cpu.reduce(&cpu_k, &cpu_refs, &[]));
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
        let spec = dense_spec(&formula, BindingOrder::LowToHigh);

        for &log_n in &[14u32, 20] {
            let n = 1usize << log_n;
            let rounds = 4;
            let mut rng = StdRng::seed_from_u64(800 + log_n as u64 + d as u64);
            let data: Vec<Vec<Fr>> = (0..n_inputs).map(|_| random_fr(&mut rng, n)).collect();
            let scalar = Fr::random(&mut rng);

            let label = format!("D{d}/2^{log_n}/{rounds}r");
            group.throughput(Throughput::Elements(n as u64));

            let mtl_k = metal.compile::<Fr>(&spec);
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<Buf<MetalBackend, Fr>> = data
                            .iter()
                            .map(|v| DeviceBuffer::Field(metal.upload(v)))
                            .collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = metal.reduce(&mtl_k, &refs, &[]);
                            for buf in &mut bufs {
                                metal.interpolate_inplace(
                                    buf.as_field_mut(),
                                    scalar,
                                    BindingOrder::LowToHigh,
                                );
                            }
                        }
                        bufs
                    });
                },
            );

            let cpu_k = cpu.compile::<Fr>(&spec);
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<Buf<CpuBackend, Fr>> = data
                            .iter()
                            .map(|v| DeviceBuffer::Field(cpu.upload(v)))
                            .collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = cpu.reduce(&cpu_k, &refs, &[]);
                            for buf in &mut bufs {
                                cpu.interpolate_inplace(
                                    buf.as_field_mut(),
                                    scalar,
                                    BindingOrder::LowToHigh,
                                );
                            }
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
                    let mut buf = metal.upload(&data);
                    metal.interpolate_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
                    buf
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("cpu/{label}"), n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let mut buf = cpu.upload(&data);
                    cpu.interpolate_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
                    buf
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
        let spec = dense_spec(&formula, BindingOrder::HighToLow);

        for &log_n in &[14u32, 20] {
            let n = 1usize << log_n;
            let rounds = 4;
            let mut rng = StdRng::seed_from_u64(800 + log_n as u64 + d as u64);
            let data: Vec<Vec<Fr>> = (0..n_inputs).map(|_| random_fr(&mut rng, n)).collect();
            let scalar = Fr::random(&mut rng);

            let label = format!("D{d}/2^{log_n}/{rounds}r");
            group.throughput(Throughput::Elements(n as u64));

            let mtl_k = metal.compile::<Fr>(&spec);
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<Buf<MetalBackend, Fr>> = data
                            .iter()
                            .map(|v| DeviceBuffer::Field(metal.upload(v)))
                            .collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = metal.reduce(&mtl_k, &refs, &[]);
                            for buf in &mut bufs {
                                metal.interpolate_inplace(
                                    buf.as_field_mut(),
                                    scalar,
                                    BindingOrder::HighToLow,
                                );
                            }
                        }
                        bufs
                    });
                },
            );

            let cpu_k = cpu.compile::<Fr>(&spec);
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{label}"), log_n),
                &log_n,
                |bench, _| {
                    bench.iter(|| {
                        let mut bufs: Vec<Buf<CpuBackend, Fr>> = data
                            .iter()
                            .map(|v| DeviceBuffer::Field(cpu.upload(v)))
                            .collect();
                        for _ in 0..rounds {
                            let refs: Vec<_> = bufs.iter().collect();
                            let _evals = cpu.reduce(&cpu_k, &refs, &[]);
                            for buf in &mut bufs {
                                cpu.interpolate_inplace(
                                    buf.as_field_mut(),
                                    scalar,
                                    BindingOrder::HighToLow,
                                );
                            }
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
        bench_reduce,
        bench_sumcheck_round,
        bench_sumcheck_round_h2l,
        bench_interpolate,
}
criterion_main!(benches);
