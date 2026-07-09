#![expect(unused_results)]
#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Field, Fr};
use jolt_kernels::cuda::{CudaKernelContext, DeviceFrVec, RoundPolyTerms};
use jolt_kernels::{bind_dense_evals_reuse, round_poly_from_factor_slices};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const LOG_SIZES: [usize; 4] = [14, 18, 20, 22];
const DEGREE: usize = 2;

fn random_factors(rng: &mut ChaCha20Rng, n: usize, degree: usize) -> Vec<Vec<Fr>> {
    (0..degree)
        .map(|_| (0..n).map(|_| Field::random(rng)).collect())
        .collect()
}

fn cpu_sumcheck(factors: &[Vec<Fr>], challenge: Fr) {
    let mut factors: Vec<Vec<Fr>> = factors.to_vec();
    let mut scratch: Vec<Vec<Fr>> = (0..factors.len()).map(|_| Vec::new()).collect();
    while factors[0].len() > 1 {
        let slices: Vec<&[Fr]> = factors.iter().map(Vec::as_slice).collect();
        let poly = round_poly_from_factor_slices(&slices, DEGREE);
        black_box(&poly);
        for (factor, scr) in factors.iter_mut().zip(&mut scratch) {
            bind_dense_evals_reuse(factor, scr, black_box(challenge));
        }
    }
}

fn cuda_sumcheck(
    ctx: &CudaKernelContext,
    factors: &mut [DeviceFrVec],
    scratch: &mut [DeviceFrVec],
    term_coeffs: &DeviceFrVec,
    offsets: &[u32],
    indices: &[u32],
    challenge: Fr,
) {
    while factors[0].len() > 1 {
        let refs: Vec<&DeviceFrVec> = factors.iter().collect();
        let coeffs = ctx
            .dense_product_round_poly(RoundPolyTerms {
                factors: &refs,
                term_coeffs,
                term_factor_offsets: offsets,
                term_factor_indices: indices,
                degree: DEGREE,
            })
            .unwrap();
        black_box(&coeffs);
        for (factor, scr) in factors.iter_mut().zip(scratch.iter_mut()) {
            ctx.bind(factor, scr, black_box(challenge)).unwrap();
        }
    }
}

fn bench_dense_sumcheck(c: &mut Criterion) {
    let ctx = CudaKernelContext::new(0).expect("cuda init");
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let challenge: Fr = Field::random(&mut rng);

    let offsets = [0u32, DEGREE as u32];
    let indices: Vec<u32> = (0..DEGREE as u32).collect();

    let mut group = c.benchmark_group("dense_sumcheck");
    group.sample_size(10);
    for &log_n in &LOG_SIZES {
        let n = 1usize << log_n;
        let factors = random_factors(&mut rng, n, DEGREE);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("cpu", log_n), &log_n, |bench, _| {
            bench.iter(|| cpu_sumcheck(&factors, challenge));
        });

        group.bench_with_input(BenchmarkId::new("cuda", log_n), &log_n, |bench, _| {
            let refs: Vec<&[Fr]> = factors.iter().map(Vec::as_slice).collect();
            let base = ctx.upload_many(&refs).unwrap();
            let term_coeffs = ctx.upload(&[Fr::from(1u64)]).unwrap();
            bench.iter(|| {
                let mut work: Vec<DeviceFrVec> =
                    base.iter().map(|f| f.try_clone().unwrap()).collect();
                let mut scratch: Vec<DeviceFrVec> =
                    (0..DEGREE).map(|_| ctx.upload(&[]).unwrap()).collect();
                cuda_sumcheck(
                    &ctx,
                    &mut work,
                    &mut scratch,
                    &term_coeffs,
                    &offsets,
                    &indices,
                    challenge,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dense_sumcheck);
criterion_main!(benches);
