#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_kernels::cuda::CudaKernelContext;
use jolt_field::{Field, Fr};
use jolt_kernels::{bind_dense_evals_reuse, bind_dense_evals_reuse_cuda, bind_dense_evals_reuse_serial};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 22];

fn random_vec(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Field::random(rng)).collect()
}

fn bench_bind_dense(c: &mut Criterion) {
    let ctx = CudaKernelContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let mut group = c.benchmark_group("bind_dense");
    for &n in &SIZES {
        let values = random_vec(&mut rng, n);
        let challenge: Fr = Field::random(&mut rng);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("serial", n), &n, |bench, _| {
            bench.iter(|| {
                let mut v = values.clone();
                let mut scratch = Vec::new();
                bind_dense_evals_reuse_serial(&mut v, &mut scratch, black_box(challenge));
                black_box(&v);
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |bench, _| {
            bench.iter(|| {
                let mut v = values.clone();
                let mut scratch = Vec::new();
                bind_dense_evals_reuse(&mut v, &mut scratch, black_box(challenge));
                black_box(&v);
            });
        });

        group.bench_with_input(BenchmarkId::new("cuda", n), &n, |bench, _| {
            let v_dev = ctx.upload(&values).unwrap();
            let mut scratch_dev = ctx.upload(&[]).unwrap();
            bench.iter(|| {
                let mut v = v_dev.try_clone().unwrap();
                bind_dense_evals_reuse_cuda(&ctx, &mut v, &mut scratch_dev, black_box(challenge))
                    .unwrap();
                black_box(&v);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bind_dense);
criterion_main!(benches);
