#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Field, Fr};
use jolt_kernels::cuda::CudaKernelContext;
use jolt_kernels::stage1::DenseOuterState;
use jolt_kernels::{CudaDenseOuterState, DenseOuterInputs};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const NUM_VARS: [usize; 4] = [18, 20, 22, 24];

const FIRST_GROUP_ROWS: [u32; 10] = [1, 2, 3, 4, 5, 6, 11, 14, 17, 18];
const SECOND_GROUP_ROWS: [u32; 9] = [0, 7, 8, 9, 10, 12, 13, 15, 16];
const ROW_COUNT: usize = 19;

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

fn cpu_construct(
    eq_evals: &[Fr],
    scale: Fr,
    weights: &[Fr],
    row_dots_a: &[Fr],
    row_dots_b: &[Fr],
) -> (Vec<Fr>, Vec<Fr>, Vec<Fr>) {
    let len = eq_evals.len();
    let cycles = len / 2;
    let mut eq = vec![Fr::from_u64(0); len];
    let mut az = vec![Fr::from_u64(0); len];
    let mut bz = vec![Fr::from_u64(0); len];
    let matvec = |rows: &[u32], cycle: usize| -> (Fr, Fr) {
        let base = cycle * ROW_COUNT;
        let mut a = Fr::from_u64(0);
        let mut b = Fr::from_u64(0);
        for (&row, &weight) in rows.iter().zip(weights.iter()) {
            a += weight * row_dots_a[base + row as usize];
            b += weight * row_dots_b[base + row as usize];
        }
        (a, b)
    };
    for cycle in 0..cycles {
        let index = cycle << 1;
        let (az0, bz0) = matvec(&FIRST_GROUP_ROWS, cycle);
        let (az1, bz1) = matvec(&SECOND_GROUP_ROWS, cycle);
        eq[index] = eq_evals[index] * scale;
        eq[index + 1] = eq_evals[index + 1] * scale;
        az[index] = az0;
        bz[index] = bz0;
        az[index + 1] = az1;
        bz[index + 1] = bz1;
    }
    (eq, az, bz)
}

fn bench_dense_outer_construct(c: &mut Criterion) {
    let ctx = CudaKernelContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(1);

    let mut group = c.benchmark_group("dense_outer_construct");
    for &num_vars in &NUM_VARS {
        let len = 1usize << num_vars;
        let cycles = len / 2;
        let eq_evals = random_vec(&mut rng, len);
        let scale: Fr = Field::random(&mut rng);
        let weights = random_vec(&mut rng, 10);
        let row_dots_a = random_vec(&mut rng, cycles * ROW_COUNT);
        let row_dots_b = random_vec(&mut rng, cycles * ROW_COUNT);
        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(BenchmarkId::new("cpu", num_vars), &num_vars, |bench, _| {
            bench.iter(|| {
                cpu_construct(
                    black_box(&eq_evals),
                    black_box(scale),
                    black_box(&weights),
                    black_box(&row_dots_a),
                    black_box(&row_dots_b),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("cuda", num_vars), &num_vars, |bench, _| {
            bench.iter(|| {
                CudaDenseOuterState::from_row_dots(
                    &ctx,
                    DenseOuterInputs {
                        eq_evals: black_box(&eq_evals),
                        scale: black_box(scale),
                        weights: black_box(&weights),
                        row_dots_a: black_box(&row_dots_a),
                        row_dots_b: black_box(&row_dots_b),
                        row_count: ROW_COUNT,
                        first_group_rows: &FIRST_GROUP_ROWS,
                        second_group_rows: &SECOND_GROUP_ROWS,
                    },
                )
                .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dense_outer, bench_dense_outer_construct);
criterion_main!(benches);
