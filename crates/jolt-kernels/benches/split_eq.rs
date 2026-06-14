#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Field, Fr};
use jolt_kernels::cuda::CudaKernelContext;
use jolt_kernels::{CudaSplitEqState, SplitEqState};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: [usize; 4] = [12, 16, 20, 22];

fn random_point(rng: &mut ChaCha20Rng, num_vars: usize) -> Vec<Fr> {
    (0..num_vars).map(|_| Field::random(rng)).collect()
}

fn cpu_bind_all(point: &[Fr], challenges: &[Fr]) -> Fr {
    let mut state = SplitEqState::<Fr>::new_low_to_high(point, None);
    for &challenge in challenges {
        state.bind(challenge);
    }
    state.eval()
}

fn cuda_bind_all(ctx: &CudaKernelContext, point: &[Fr], challenges: &[Fr]) -> Fr {
    let mut state = CudaSplitEqState::new_low_to_high(ctx, point, None).unwrap();
    for &challenge in challenges {
        state.bind(challenge).unwrap();
    }
    state.eval().unwrap()
}

fn bench_split_eq(c: &mut Criterion) {
    let ctx = CudaKernelContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let mut group = c.benchmark_group("split_eq_bind_all");
    for &num_vars in &NUM_VARS {
        let point = random_point(&mut rng, num_vars);
        let challenges = random_point(&mut rng, num_vars);
        group.throughput(Throughput::Elements(1u64 << num_vars));

        group.bench_with_input(BenchmarkId::new("cpu", num_vars), &num_vars, |bench, _| {
            bench.iter(|| cpu_bind_all(black_box(&point), black_box(&challenges)));
        });
        group.bench_with_input(BenchmarkId::new("cuda", num_vars), &num_vars, |bench, _| {
            bench.iter(|| cuda_bind_all(&ctx, black_box(&point), black_box(&challenges)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_split_eq);
criterion_main!(benches);
