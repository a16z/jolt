#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::arkworks::cuda::{CudaFieldContext, DeviceFrVec};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const SIZES: [usize; 4] = [1 << 12, 1 << 16, 1 << 20, 1 << 22];

fn random_vec(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Field::random(rng)).collect()
}

fn bench_map(c: &mut Criterion) {
    let ctx = CudaFieldContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    for (name, gpu, cpu) in [
        (
            "add",
            (&|c: &CudaFieldContext, a: &mut DeviceFrVec, b: &DeviceFrVec| c.add(a, b).unwrap())
                as &dyn Fn(&CudaFieldContext, &mut DeviceFrVec, &DeviceFrVec),
            (&|a: &[Fr], b: &[Fr]| a.iter().zip(b).map(|(x, y)| *x + *y).collect::<Vec<_>>())
                as &dyn Fn(&[Fr], &[Fr]) -> Vec<Fr>,
        ),
        (
            "sub",
            &|c, a, b| c.sub(a, b).unwrap(),
            &|a, b| a.iter().zip(b).map(|(x, y)| *x - *y).collect(),
        ),
        (
            "mul",
            &|c, a, b| c.mul(a, b).unwrap(),
            &|a, b| a.iter().zip(b).map(|(x, y)| *x * *y).collect(),
        ),
    ] {
        let mut group = c.benchmark_group(format!("map/{name}"));
        for &n in &SIZES {
            let a = random_vec(&mut rng, n);
            let b = random_vec(&mut rng, n);
            let mut a_dev = ctx.upload(&a).unwrap();
            let b_dev = ctx.upload(&b).unwrap();
            group.throughput(Throughput::Elements(n as u64));

            group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
                bench.iter(|| cpu(black_box(&a), black_box(&b)));
            });
            group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
                bench.iter(|| gpu(&ctx, &mut a_dev, black_box(&b_dev)));
            });
        }
        group.finish();
    }
}

fn bench_reduce(c: &mut Criterion) {
    let ctx = CudaFieldContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(1);

    for (name, gpu, cpu) in [
        (
            "sum",
            (&|c: &CudaFieldContext, a: &DeviceFrVec| c.sum(a).unwrap())
                as &dyn Fn(&CudaFieldContext, &DeviceFrVec) -> Fr,
            (&|a: &[Fr]| a.iter().copied().sum::<Fr>()) as &dyn Fn(&[Fr]) -> Fr,
        ),
        (
            "product",
            &|c, a| c.product(a).unwrap(),
            &|a| a.iter().copied().product(),
        ),
    ] {
        let mut group = c.benchmark_group(format!("reduce/{name}"));
        for &n in &SIZES {
            let a = random_vec(&mut rng, n);
            let a_dev = ctx.upload(&a).unwrap();
            group.throughput(Throughput::Elements(n as u64));

            group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
                bench.iter(|| cpu(black_box(&a)));
            });
            group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
                bench.iter(|| gpu(&ctx, black_box(&a_dev)));
            });
        }
        group.finish();
    }
}

const CHAIN_STEPS: usize = 8;

fn cpu_chain(a: &[Fr], b: &[Fr], steps: usize) -> Vec<Fr> {
    let mut acc: Vec<Fr> = a.to_vec();
    for _ in 0..steps {
        acc = acc.iter().zip(b).map(|(x, y)| *x * *y).collect();
        acc = acc.iter().zip(b).map(|(x, y)| *x + *y).collect();
    }
    acc
}

fn gpu_chain(ctx: &CudaFieldContext, a: &[Fr], b: &[Fr], steps: usize) -> Vec<Fr> {
    let mut acc = ctx.upload(a).unwrap();
    let b_dev = ctx.upload(b).unwrap();
    for _ in 0..steps {
        ctx.mul(&mut acc, &b_dev).unwrap();
        ctx.add(&mut acc, &b_dev).unwrap();
    }
    acc.to_host().unwrap()
}

fn gpu_chain_fma(ctx: &CudaFieldContext, a: &[Fr], b: &[Fr], steps: usize) -> Vec<Fr> {
    let mut acc = ctx.upload(a).unwrap();
    let b_dev = ctx.upload(b).unwrap();
    for _ in 0..steps {
        ctx.fma(&mut acc, &b_dev, &b_dev).unwrap();
    }
    acc.to_host().unwrap()
}

fn bench_chain(c: &mut Criterion) {
    let ctx = CudaFieldContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(2);

    let mut group = c.benchmark_group("chain");
    for &n in &SIZES {
        let a = random_vec(&mut rng, n);
        let b = random_vec(&mut rng, n);
        group.throughput(Throughput::Elements((n * CHAIN_STEPS * 2) as u64));

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
            bench.iter(|| cpu_chain(black_box(&a), black_box(&b), CHAIN_STEPS));
        });
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
            bench.iter(|| gpu_chain(&ctx, black_box(&a), black_box(&b), CHAIN_STEPS));
        });
    }
    group.finish();
}

fn bench_chain_fma(c: &mut Criterion) {
    let ctx = CudaFieldContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(2);

    let mut group = c.benchmark_group("chain_fma");
    for &n in &SIZES {
        let a = random_vec(&mut rng, n);
        let b = random_vec(&mut rng, n);
        group.throughput(Throughput::Elements((n * CHAIN_STEPS * 2) as u64));

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
            bench.iter(|| cpu_chain(black_box(&a), black_box(&b), CHAIN_STEPS));
        });
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
            bench.iter(|| gpu_chain_fma(&ctx, black_box(&a), black_box(&b), CHAIN_STEPS));
        });
    }
    group.finish();
}

fn cpu_reduce_chain(a: &[Fr], b: &[Fr], steps: usize) -> Fr {
    let mut acc = a.to_vec();
    let mut total = Fr::from_u64(0);
    for _ in 0..steps {
        acc = acc.iter().zip(b).map(|(x, y)| *x * *y).collect();
        total += acc.iter().copied().sum::<Fr>();
    }
    total
}

fn gpu_reduce_chain(ctx: &CudaFieldContext, a: &[Fr], b: &[Fr], steps: usize) -> Fr {
    let mut acc = ctx.upload(a).unwrap();
    let b_dev = ctx.upload(b).unwrap();
    let mut total = Fr::from_u64(0);
    for _ in 0..steps {
        ctx.mul(&mut acc, &b_dev).unwrap();
        total += ctx.sum(&acc).unwrap();
    }
    total
}

fn bench_reduce_chain(c: &mut Criterion) {
    let ctx = CudaFieldContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(3);

    let mut group = c.benchmark_group("reduce_chain");
    for &n in &SIZES {
        let a = random_vec(&mut rng, n);
        let b = random_vec(&mut rng, n);
        group.throughput(Throughput::Elements((n * CHAIN_STEPS * 2) as u64));

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
            bench.iter(|| cpu_reduce_chain(black_box(&a), black_box(&b), CHAIN_STEPS));
        });
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
            bench.iter(|| gpu_reduce_chain(&ctx, black_box(&a), black_box(&b), CHAIN_STEPS));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_map,
    bench_reduce,
    bench_chain,
    bench_chain_fma,
    bench_reduce_chain
);
criterion_main!(benches);
