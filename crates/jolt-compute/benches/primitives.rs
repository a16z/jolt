#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compute::{ComputeBackend, CpuBackend, CpuKernel};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_field_vec(n: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

fn make_product_sum_kernel(d: usize) -> CpuKernel<Fr> {
    CpuKernel::new(move |lo: &[Fr], hi: &[Fr], out: &mut [Fr]| {
        for (t, slot) in out.iter_mut().enumerate() {
            let t_f = Fr::from_u64(t as u64);
            let mut product = Fr::from_u64(1);
            for j in 0..d {
                product *= lo[j] + t_f * (hi[j] - lo[j]);
            }
            *slot = product;
        }
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

fn bench_interpolate_pairs_inplace(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("interpolate_pairs_inplace");

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let half = (n / 2) as u64;
        let data = random_field_vec(n, 42);
        let scalar = Fr::from_u64(0x1234_5678);

        group.throughput(Throughput::Elements(half));

        group.bench_with_input(
            BenchmarkId::new("allocating", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let buf = data.clone();
                    black_box(backend.interpolate_pairs::<Fr, Fr>(buf, scalar));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("inplace_low_to_high", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut buf = data.clone();
                    backend.interpolate_pairs_inplace(
                        &mut buf,
                        scalar,
                        jolt_compute::BindingOrder::LowToHigh,
                    );
                    black_box(&buf);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("inplace_high_to_low", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut buf = data.clone();
                    backend.interpolate_pairs_inplace(
                        &mut buf,
                        scalar,
                        jolt_compute::BindingOrder::HighToLow,
                    );
                    black_box(&buf);
                });
            },
        );
    }

    group.finish();
}

fn bench_pairwise_reduce_fixed(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce_fixed");

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

            group.throughput(Throughput::Elements(half as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("dynamic/D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| {
                        black_box(backend.pairwise_reduce(&input_refs, &weights, &kernel, d));
                    });
                },
            );

            match d {
                4 => {
                    group.bench_with_input(
                        BenchmarkId::new(format!("fixed/D={d}"), format!("2^{log_n}")),
                        &n,
                        |b, _| {
                            b.iter(|| {
                                black_box(
                                    backend
                                        .pairwise_reduce_fixed::<Fr, 4>(&input_refs, &weights, &kernel),
                                );
                            });
                        },
                    );
                }
                8 => {
                    group.bench_with_input(
                        BenchmarkId::new(format!("fixed/D={d}"), format!("2^{log_n}")),
                        &n,
                        |b, _| {
                            b.iter(|| {
                                black_box(
                                    backend
                                        .pairwise_reduce_fixed::<Fr, 8>(&input_refs, &weights, &kernel),
                                );
                            });
                        },
                    );
                }
                16 => {
                    group.bench_with_input(
                        BenchmarkId::new(format!("fixed/D={d}"), format!("2^{log_n}")),
                        &n,
                        |b, _| {
                            b.iter(|| {
                                black_box(
                                    backend.pairwise_reduce_fixed::<Fr, 16>(
                                        &input_refs,
                                        &weights,
                                        &kernel,
                                    ),
                                );
                            });
                        },
                    );
                }
                _ => {}
            }
        }
    }

    group.finish();
}

fn bench_tensor_pairwise_reduce(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("tensor_pairwise_reduce");

    let d = 4;
    let kernel = make_product_sum_kernel(d);

    // Test various outer×inner splits at total ~2^18 pairs
    for (outer_log, inner_log) in [(5, 13), (9, 9), (13, 5)] {
        let outer_len = 1usize << outer_log;
        let inner_len = 1usize << inner_log;
        let total_pairs = outer_len * inner_len;
        let n = total_pairs * 2;

        let inputs: Vec<Vec<Fr>> = (0..d)
            .map(|i| random_field_vec(n, 100 + i as u64))
            .collect();
        let input_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
        let outer_w = random_field_vec(outer_len, 200);
        let inner_w = random_field_vec(inner_len, 201);

        // Flat weights for comparison
        let mut flat_w = Vec::with_capacity(total_pairs);
        for &o in &outer_w {
            for &i in &inner_w {
                flat_w.push(o * i);
            }
        }

        group.throughput(Throughput::Elements(total_pairs as u64));

        let label = format!("{outer_log}+{inner_log}");

        group.bench_with_input(
            BenchmarkId::new("flat", &label),
            &total_pairs,
            |b, _| {
                b.iter(|| {
                    black_box(backend.pairwise_reduce(&input_refs, &flat_w, &kernel, d));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tensor", &label),
            &total_pairs,
            |b, _| {
                b.iter(|| {
                    black_box(backend.tensor_pairwise_reduce(
                        &input_refs,
                        &outer_w,
                        &inner_w,
                        &kernel,
                        d,
                    ));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tensor_fixed", &label),
            &total_pairs,
            |b, _| {
                b.iter(|| {
                    black_box(backend.tensor_pairwise_reduce_fixed::<Fr, 4>(
                        &input_refs,
                        &outer_w,
                        &inner_w,
                        &kernel,
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_pairwise_reduce_multi(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("pairwise_reduce_multi");

    let k1 = make_product_sum_kernel(4);
    let k2 = make_product_sum_kernel(4);

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let half = n / 2;

        let inputs: Vec<Vec<Fr>> = (0..4)
            .map(|i| random_field_vec(n, 100 + i as u64))
            .collect();
        let input_refs: Vec<&Vec<Fr>> = inputs.iter().collect();
        let weights = random_field_vec(half, 200);

        group.throughput(Throughput::Elements(half as u64));

        group.bench_with_input(
            BenchmarkId::new("individual_2x", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(backend.pairwise_reduce(&input_refs, &weights, &k1, 4));
                    black_box(backend.pairwise_reduce(&input_refs, &weights, &k2, 4));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("multi_2x", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(backend.pairwise_reduce_multi(
                        &input_refs,
                        &weights,
                        &[(&k1, 4), (&k2, 4)],
                    ));
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
    bench_pairwise_reduce,
    bench_product_table,
    bench_interpolate_pairs_inplace,
    bench_pairwise_reduce_fixed,
    bench_tensor_pairwise_reduce,
    bench_pairwise_reduce_multi,
);
criterion_main!(benches);
