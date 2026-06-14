#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Field, Fr};
use jolt_kernels::cuda::CudaKernelContext;
use jolt_kernels::stage1::DenseOuterState;
use jolt_kernels::CudaDenseOuterState;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: [usize; 4] = [18, 20, 22, 24];

fn random_vec(rng: &mut ChaCha20Rng, n: usize) -> Vec<Fr> {
    (0..n).map(|_| Field::random(rng)).collect()
}

fn challenges(num_vars: usize) -> Vec<Fr> {
    (0..num_vars).map(|i| Fr::from_u64((i + 1) as u64)).collect()
}

fn cpu_prove(eq: &[Fr], az: &[Fr], bz: &[Fr], challenges: &[Fr]) -> Fr {
    let mut state = DenseOuterState::from_raw(eq.to_vec(), az.to_vec(), bz.to_vec());
    let mut acc = Fr::from_u64(0);
    for &challenge in challenges {
        acc += state.round_poly().coefficients()[0];
        state.bind(challenge);
    }
    acc
}

fn cuda_prove(
    ctx: &CudaKernelContext,
    eq: &[Fr],
    az: &[Fr],
    bz: &[Fr],
    challenges: &[Fr],
) -> Fr {
    let mut state = CudaDenseOuterState::from_host(ctx, eq, az, bz).unwrap();
    let mut acc = Fr::from_u64(0);
    for &challenge in challenges {
        acc += state.round_poly().unwrap().coefficients()[0];
        state.bind(challenge).unwrap();
    }
    acc
}

fn bench_dense_outer(c: &mut Criterion) {
    let ctx = CudaKernelContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let mut group = c.benchmark_group("dense_outer_prove");
    for &num_vars in &NUM_VARS {
        let len = 1usize << num_vars;
        let eq = random_vec(&mut rng, len);
        let az = random_vec(&mut rng, len);
        let bz = random_vec(&mut rng, len);
        let challenges = challenges(num_vars);
        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(BenchmarkId::new("cpu", num_vars), &num_vars, |bench, _| {
            bench.iter(|| {
                cpu_prove(
                    black_box(&eq),
                    black_box(&az),
                    black_box(&bz),
                    black_box(&challenges),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("cuda", num_vars), &num_vars, |bench, _| {
            bench.iter(|| {
                cuda_prove(
                    &ctx,
                    black_box(&eq),
                    black_box(&az),
                    black_box(&bz),
                    black_box(&challenges),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dense_outer);
criterion_main!(benches);
