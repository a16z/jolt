#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compute::{AnyBuffer, ComputeBackend, CpuBackend, CpuKernel};
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
        let half = (n / 2) as u64;
        let data = random_field_vec(n, 42);
        let scalar = Fr::from_u64(0x1234_5678);

        group.throughput(Throughput::Elements(half));

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
        let half = (n / 2) as u64;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let data: Vec<u8> = (0..n).map(|_| rand::Rng::gen(&mut rng)).collect();
        let scalar = Fr::from_u64(0x1234_5678);

        group.throughput(Throughput::Elements(half));

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

            // Throughput = number of pairs reduced
            group.throughput(Throughput::Elements(half as u64));

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

fn bench_pairwise_reduce_mixed(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce_mixed");

    // Compare: D=4 inputs where half are u8 (compact) vs all Fr (promoted)
    let d = 4;
    let kernel = make_product_sum_kernel(d);

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let half = n / 2;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        // 2 compact u8 inputs + 2 field inputs
        let compact_a: Vec<u8> = (0..n).map(|_| rand::Rng::gen(&mut rng)).collect();
        let compact_b: Vec<u8> = (0..n).map(|_| rand::Rng::gen(&mut rng)).collect();
        let field_a = random_field_vec(n, 100);
        let field_b = random_field_vec(n, 101);
        let weights = random_field_vec(half, 200);

        group.throughput(Throughput::Elements(half as u64));

        // Mixed path: reads u8 directly
        group.bench_with_input(
            BenchmarkId::new("mixed", format!("2^{log_n}")),
            &n,
            |b, _| {
                let inputs: Vec<AnyBuffer<'_, Fr>> = vec![
                    AnyBuffer::from(compact_a.as_slice()),
                    AnyBuffer::from(compact_b.as_slice()),
                    AnyBuffer::field(field_a.as_slice()),
                    AnyBuffer::field(field_b.as_slice()),
                ];
                b.iter(|| {
                    black_box(backend.pairwise_reduce_mixed(&inputs, &weights, &kernel, d));
                });
            },
        );

        // Promoted path: pre-convert u8 to Fr, then use standard pairwise_reduce
        let promoted_a: Vec<Fr> = compact_a.iter().map(|&x| Fr::from_u8(x)).collect();
        let promoted_b: Vec<Fr> = compact_b.iter().map(|&x| Fr::from_u8(x)).collect();
        group.bench_with_input(
            BenchmarkId::new("promoted", format!("2^{log_n}")),
            &n,
            |b, _| {
                let input_refs: Vec<&Vec<Fr>> =
                    vec![&promoted_a, &promoted_b, &field_a, &field_b];
                b.iter(|| {
                    black_box(backend.pairwise_reduce(&input_refs, &weights, &kernel, d));
                });
            },
        );
    }

    group.finish();
}

fn bench_interpolate_pairs_batch(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("interpolate_pairs_batch");

    for (n_bufs, log_n) in [(8, 18), (32, 16), (128, 14)] {
        let n = 1usize << log_n;
        let total_pairs = (n_bufs * n / 2) as u64;
        let scalar = Fr::from_u64(0x1234_5678);

        let bufs: Vec<Vec<Fr>> = (0..n_bufs)
            .map(|i| random_field_vec(n, 300 + i as u64))
            .collect();

        group.throughput(Throughput::Elements(total_pairs));

        group.bench_with_input(
            BenchmarkId::new("batch", format!("{n_bufs}x2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let cloned: Vec<Vec<Fr>> = bufs.clone();
                    black_box(backend.interpolate_pairs_batch(cloned, scalar));
                });
            },
        );
    }

    group.finish();
}

fn bench_interpolate_mixed(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("interpolate_mixed");

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let half = (n / 2) as u64;
        let scalar = Fr::from_u64(0x1234_5678);

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let data_u8: Vec<u8> = (0..n).map(|_| rand::Rng::gen(&mut rng)).collect();
        let data_fr = random_field_vec(n, 100);

        group.throughput(Throughput::Elements(half));

        group.bench_with_input(
            BenchmarkId::new("u8", format!("2^{log_n}")),
            &n,
            |b, _| {
                let buf = AnyBuffer::from(data_u8.as_slice());
                b.iter(|| {
                    black_box(backend.interpolate_mixed::<Fr>(&buf, scalar));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("field", format!("2^{log_n}")),
            &n,
            |b, _| {
                let buf = AnyBuffer::field(data_fr.as_slice());
                b.iter(|| {
                    black_box(backend.interpolate_mixed::<Fr>(&buf, scalar));
                });
            },
        );
    }

    group.finish();
}

fn bench_product_table(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("product_table");

    for n_vars in [16, 20, 24] {
        let table_size = 1u64 << n_vars;
        let point = random_field_vec(n_vars, 42);

        group.throughput(Throughput::Elements(table_size));

        group.bench_with_input(
            BenchmarkId::new("compute", format!("n={n_vars}")),
            &n_vars,
            |b, _| {
                b.iter(|| {
                    black_box(backend.product_table(&point));
                });
            },
        );

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
    bench_interpolate_pairs_batch,
    bench_interpolate_mixed,
    bench_pairwise_reduce,
    bench_pairwise_reduce_mixed,
    bench_product_table,
);
criterion_main!(benches);
