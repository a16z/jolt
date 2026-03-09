//! Benchmarks comparing Metal Fr arithmetic throughput vs CPU.
//!
//! Measures end-to-end latency including buffer creation (effectively free
//! on Apple Silicon unified memory). The meaningful comparison is throughput
//! at large array sizes where Metal parallelism saturates.

#![cfg(target_os = "macos")]
#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compute::{BindingOrder, ComputeBackend, CpuBackend};
use jolt_field::Field;
use jolt_field::Fr;
use jolt_ir::{KernelDescriptor, KernelShape};
use jolt_metal::field::{dispatch_binary, dispatch_fmadd, dispatch_unary, FrKernels, MetalFr};
use jolt_metal::MetalBackend;
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

fn random_fr(rng: &mut StdRng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Fr::random(rng)).collect()
}

fn bench_kernel_compile(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let mut group = c.benchmark_group("kernel_compile");
    group.sample_size(20);

    for &d in &[4usize, 8, 16] {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: d,
                num_products: 1,
            },
            degree: d,
            tensor_split: None,
        };

        group.bench_with_input(BenchmarkId::new("metal_product_sum", d), &d, |bench, _| {
            bench.iter(|| metal.compile_kernel::<Fr>(&desc));
        });

        group.bench_with_input(BenchmarkId::new("cpu_product_sum", d), &d, |bench, _| {
            bench.iter(|| jolt_cpu_kernels::compile::<Fr>(&desc));
        });
    }
    group.finish();
}

fn bench_pairwise_reduce(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce");
    group.sample_size(20);

    for &d in &[4usize, 8] {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: d,
                num_products: 1,
            },
            degree: d,
            tensor_split: None,
        };

        let cpu_k = jolt_cpu_kernels::compile::<Fr>(&desc);
        let mtl_k = metal.compile_kernel::<Fr>(&desc);

        for &n in &[1 << 14, 1 << 18, 1 << 24] {
            let mut rng = StdRng::seed_from_u64(100 + n as u64);
            let inputs: Vec<Vec<Fr>> = (0..d).map(|_| random_fr(&mut rng, n)).collect();
            let weights = random_fr(&mut rng, n / 2);

            let mtl_bufs: Vec<_> = inputs.iter().map(|v| metal.upload(v)).collect();
            let mtl_w = metal.upload(&weights);
            let cpu_w = cpu.upload(&weights);

            let label = format!("D{d}/n{n}");

            group.throughput(Throughput::Elements((n / 2) as u64));

            let mtl_refs: Vec<_> = mtl_bufs.iter().collect();
            group.bench_with_input(
                BenchmarkId::new(format!("metal/{label}"), n),
                &n,
                |bench, _| {
                    bench.iter(|| {
                        metal.pairwise_reduce(
                            &mtl_refs,
                            &mtl_w,
                            &mtl_k,
                            desc.num_evals(),
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
                            &cpu_w,
                            &cpu_k,
                            desc.num_evals(),
                            BindingOrder::LowToHigh,
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_product_table(c: &mut Criterion) {
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let mut group = c.benchmark_group("product_table");
    group.sample_size(20);

    for &n_vars in &[12usize, 16, 20] {
        let mut rng = StdRng::seed_from_u64(200 + n_vars as u64);
        let point: Vec<Fr> = (0..n_vars).map(|_| Fr::random(&mut rng)).collect();

        group.throughput(Throughput::Elements(1u64 << n_vars));

        group.bench_with_input(BenchmarkId::new("metal", n_vars), &n_vars, |bench, _| {
            bench.iter(|| metal.product_table::<Fr>(&point));
        });

        group.bench_with_input(BenchmarkId::new("cpu", n_vars), &n_vars, |bench, _| {
            bench.iter(|| cpu.product_table::<Fr>(&point));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mul,
    bench_add,
    bench_sqr,
    bench_fmadd,
    bench_kernel_compile,
    bench_pairwise_reduce,
    bench_product_table,
);
criterion_main!(benches);
