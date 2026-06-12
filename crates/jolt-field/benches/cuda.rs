#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::arkworks::cuda::CudaFieldContext;
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
            (&|c: &CudaFieldContext, a: &[Fr], b: &[Fr]| c.add(a, b).unwrap())
                as &dyn Fn(&CudaFieldContext, &[Fr], &[Fr]) -> Vec<Fr>,
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
            group.throughput(Throughput::Elements(n as u64));

            group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
                bench.iter(|| cpu(black_box(&a), black_box(&b)));
            });
            group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
                bench.iter(|| gpu(&ctx, black_box(&a), black_box(&b)));
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
            (&|c: &CudaFieldContext, a: &[Fr]| c.sum(a).unwrap())
                as &dyn Fn(&CudaFieldContext, &[Fr]) -> Fr,
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
            group.throughput(Throughput::Elements(n as u64));

            group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bench, _| {
                bench.iter(|| cpu(black_box(&a)));
            });
            group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bench, _| {
                bench.iter(|| gpu(&ctx, black_box(&a)));
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_map, bench_reduce);
criterion_main!(benches);
