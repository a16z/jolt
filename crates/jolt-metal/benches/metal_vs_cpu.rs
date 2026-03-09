//! Benchmarks comparing Metal Fr arithmetic throughput vs CPU.
//!
//! Measures end-to-end latency including buffer creation (effectively free
//! on Apple Silicon unified memory). The meaningful comparison is throughput
//! at large array sizes where Metal parallelism saturates.

#![cfg(target_os = "macos")]
#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::Field;
use jolt_field::Fr;
use jolt_metal::field::{dispatch_binary, dispatch_fmadd, dispatch_unary, FrKernels, MetalFr};
use num_traits::Zero;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn to_metal(f: Fr) -> MetalFr {
    MetalFr::from_u64_limbs(f.inner_limbs().0)
}

struct BenchCtx {
    device: metal::Device,
    queue: metal::CommandQueue,
    kernels: FrKernels,
}

impl BenchCtx {
    fn new() -> Self {
        let device = metal::Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let kernels = FrKernels::new(&device);
        Self {
            device,
            queue,
            kernels,
        }
    }
}

fn random_metal_elements(rng: &mut StdRng, n: usize) -> (Vec<Fr>, Vec<MetalFr>) {
    let cpu: Vec<Fr> = (0..n).map(|_| Fr::random(rng)).collect();
    let mtl: Vec<MetalFr> = cpu.iter().map(|x| to_metal(*x)).collect();
    (cpu, mtl)
}

const SIZES: [usize; 3] = [1 << 12, 1 << 16, 1 << 20];

fn bench_mul(c: &mut Criterion) {
    let ctx = BenchCtx::new();
    let mut group = c.benchmark_group("fr_mul");

    for &size in &SIZES {
        let mut rng = StdRng::seed_from_u64(42);
        let (a_cpu, a_mtl) = random_metal_elements(&mut rng, size);
        let (b_cpu, b_mtl) = random_metal_elements(&mut rng, size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("metal", size), &size, |bench, _| {
            bench.iter(|| {
                dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.mul, &a_mtl, &b_mtl)
            });
        });

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |bench, _| {
            bench.iter(|| {
                a_cpu
                    .iter()
                    .zip(b_cpu.iter())
                    .map(|(a, b)| *a * *b)
                    .collect::<Vec<_>>()
            });
        });
    }
    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let ctx = BenchCtx::new();
    let mut group = c.benchmark_group("fr_add");

    for &size in &SIZES {
        let mut rng = StdRng::seed_from_u64(43);
        let (a_cpu, a_mtl) = random_metal_elements(&mut rng, size);
        let (b_cpu, b_mtl) = random_metal_elements(&mut rng, size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("metal", size), &size, |bench, _| {
            bench.iter(|| {
                dispatch_binary(&ctx.device, &ctx.queue, &ctx.kernels.add, &a_mtl, &b_mtl)
            });
        });

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |bench, _| {
            bench.iter(|| {
                a_cpu
                    .iter()
                    .zip(b_cpu.iter())
                    .map(|(a, b)| *a + *b)
                    .collect::<Vec<_>>()
            });
        });
    }
    group.finish();
}

fn bench_sqr(c: &mut Criterion) {
    let ctx = BenchCtx::new();
    let mut group = c.benchmark_group("fr_sqr");

    for &size in &SIZES {
        let mut rng = StdRng::seed_from_u64(44);
        let (a_cpu, a_mtl) = random_metal_elements(&mut rng, size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("metal", size), &size, |bench, _| {
            bench.iter(|| dispatch_unary(&ctx.device, &ctx.queue, &ctx.kernels.sqr, &a_mtl));
        });

        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |bench, _| {
            bench.iter(|| a_cpu.iter().map(|a| a.square()).collect::<Vec<_>>());
        });
    }
    group.finish();
}

fn bench_fmadd(c: &mut Criterion) {
    let ctx = BenchCtx::new();
    let mut group = c.benchmark_group("fr_fmadd");
    group.sample_size(20);

    let stride = 1 << 16;
    let mut rng = StdRng::seed_from_u64(45);
    let (a_cpu, a_mtl) = random_metal_elements(&mut rng, stride);
    let (b_cpu, b_mtl) = random_metal_elements(&mut rng, stride);

    for &n_threads in &[1 << 10, 1 << 14, 1 << 18] {
        group.throughput(Throughput::Elements((n_threads * 256) as u64));

        group.bench_with_input(
            BenchmarkId::new("metal", n_threads),
            &n_threads,
            |bench, &nt| {
                bench.iter(|| {
                    dispatch_fmadd(
                        &ctx.device,
                        &ctx.queue,
                        &ctx.kernels.fmadd,
                        &a_mtl,
                        &b_mtl,
                        nt,
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpu", n_threads),
            &n_threads,
            |bench, &nt| {
                bench.iter(|| {
                    (0..nt)
                        .map(|tid| {
                            let base = tid * 256;
                            let mut acc = Fr::zero();
                            for i in 0..256 {
                                let idx = (base + i) % stride;
                                acc += a_cpu[idx] * b_cpu[idx];
                            }
                            acc
                        })
                        .collect::<Vec<_>>()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mul, bench_add, bench_sqr, bench_fmadd);
criterion_main!(benches);
