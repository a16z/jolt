#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_compute::{ComputeBackend, CpuBackend, CpuKernel};
use jolt_field::{Field, Fr};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_field_vec(n: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

fn make_product_sum_kernel(d: usize) -> CpuKernel<Fr> {
    CpuKernel::new(move |lo: &[Fr], hi: &[Fr], degree: usize| {
        let mut evals = vec![Fr::zero(); degree + 1];
        for (t, eval) in evals.iter_mut().enumerate() {
            let t_f = Fr::from_u64(t as u64);
            let mut product = Fr::from_u64(1);
            for j in 0..d {
                product *= lo[j] + t_f * (hi[j] - lo[j]);
            }
            *eval += product;
        }
        evals
    })
}

fn bench_interpolate_pairs(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("interpolate_pairs");

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let data = random_field_vec(n, 42);
        let scalar = Fr::from_u64(0x1234_5678);

        group.bench_with_input(
            BenchmarkId::new("field", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let buf = backend.upload(&data);
                    black_box(backend.interpolate_pairs::<Fr, Fr>(buf, scalar));
                });
            },
        );

        // Compare against jolt-poly Polynomial::bind
        group.bench_with_input(
            BenchmarkId::new("poly_bind", format!("2^{log_n}")),
            &n,
            |b, _| {
                let poly = jolt_poly::Polynomial::new(data.clone());
                b.iter(|| {
                    let mut p = poly.clone();
                    p.bind(black_box(scalar));
                    black_box(&p);
                });
            },
        );
    }

    group.finish();
}

fn bench_interpolate_pairs_compact(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("interpolate_pairs_compact");

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let data: Vec<u8> = (0..n).map(|_| rand::Rng::gen(&mut rng)).collect();
        let scalar = Fr::from_u64(0x1234_5678);

        group.bench_with_input(BenchmarkId::new("u8", format!("2^{log_n}")), &n, |b, _| {
            b.iter(|| {
                let buf = backend.upload(&data);
                black_box(backend.interpolate_pairs::<u8, Fr>(buf, scalar));
            });
        });
    }

    group.finish();
}

fn bench_pairwise_reduce(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce");

    for d in [4, 8, 16] {
        let kernel = make_product_sum_kernel(d);

        for log_n in [16, 18, 20] {
            let n = 1usize << log_n;
            let half = n / 2;

            let inputs: Vec<Vec<Fr>> = (0..d)
                .map(|i| random_field_vec(n, 100 + i as u64))
                .collect();
            let input_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
            let weights = random_field_vec(half, 200);

            group.bench_with_input(
                BenchmarkId::new(format!("D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| {
                        black_box(backend.pairwise_reduce(&input_refs, &weights, &kernel, d));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_product_table(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("product_table");

    for n_vars in [16, 20, 24] {
        let point = random_field_vec(n_vars, 42);

        group.bench_with_input(
            BenchmarkId::new("compute", format!("n={n_vars}")),
            &n_vars,
            |b, _| {
                b.iter(|| {
                    black_box(backend.product_table(&point));
                });
            },
        );

        // Compare against jolt-poly EqPolynomial::evaluations
        group.bench_with_input(
            BenchmarkId::new("eq_poly", format!("n={n_vars}")),
            &n_vars,
            |b, _| {
                let eq = jolt_poly::EqPolynomial::new(point.clone());
                b.iter(|| {
                    black_box(eq.evaluations());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_interpolate_pairs,
    bench_interpolate_pairs_compact,
    bench_pairwise_reduce,
    bench_product_table,
);
criterion_main!(benches);
