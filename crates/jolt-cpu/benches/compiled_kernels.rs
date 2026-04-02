#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{BindingOrder, Factor, Formula, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::{compile, CpuBackend};
use jolt_field::{Field, Fr};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_vecs(n: usize, seed: u64) -> (Vec<Fr>, Vec<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let lo: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let hi: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    (lo, hi)
}

fn random_field_vec(n: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

fn bench_product_sum_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_sum_kernel_eval");

    for d in [4, 8, 16] {
        for num_products in [1, 4] {
            let total_inputs = d * num_products;
            let (lo, hi) = random_vecs(total_inputs, d as u64 * 100 + num_products as u64);

            let formula = product_sum_formula(d, num_products);
            let num_evals = formula.degree();
            let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
            let kernel = compile::<Fr>(&spec);

            group.throughput(Throughput::Elements(total_inputs as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("D={d}/P={num_products}"), "specialized"),
                &d,
                |b, _| {
                    b.iter(|| {
                        let mut out = vec![Fr::zero(); num_evals];
                        kernel.evaluate(&lo, &hi, &[], &mut out);
                        black_box(&out);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_custom_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("custom_kernel_eval");

    // Booleanity: h^2 - h = [coeff:1 Input(0)*Input(0)] + [coeff:-1 Input(0)]
    let formula = Formula::from_terms(vec![
        ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(0)],
        },
        ProductTerm {
            coefficient: -1,
            factors: vec![Factor::Input(0)],
        },
    ]);
    let num_evals = formula.degree();
    let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
    let kernel = compile::<Fr>(&spec);
    let (lo, hi) = random_vecs(1, 999);

    group.throughput(Throughput::Elements(1));
    group.bench_function("booleanity", |bench| {
        bench.iter(|| {
            let mut out = vec![Fr::zero(); num_evals];
            kernel.evaluate(&lo, &hi, &[], &mut out);
            black_box(&out);
        });
    });

    // Product: Input(0) * Input(1) * Input(2) * Input(3)
    let formula = product_sum_formula(4, 1);
    let num_evals = formula.degree();
    let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
    let kernel = compile::<Fr>(&spec);
    let (lo, hi) = random_vecs(4, 1000);

    group.throughput(Throughput::Elements(4));
    group.bench_function("product_4_via_custom", |bench| {
        bench.iter(|| {
            let mut out = vec![Fr::zero(); num_evals];
            kernel.evaluate(&lo, &hi, &[], &mut out);
            black_box(&out);
        });
    });

    group.finish();
}

fn bench_dense_reduce(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("dense_reduce");

    for d in [4, 8, 16] {
        let formula = product_sum_formula(d, 1);
        let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
        let kernel = compile::<Fr>(&spec);

        for log_n in [14, 18] {
            let n = 1usize << log_n;
            let bufs: Vec<Buf<CpuBackend, Fr>> = (0..d)
                .map(|i| DeviceBuffer::Field(random_field_vec(n, 500 + i as u64)))
                .collect();
            let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();

            group.throughput(Throughput::Elements((n / 2) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| black_box(backend.reduce(&kernel, &buf_refs, &[])));
                },
            );
        }
    }

    group.finish();
}

fn bench_sparse_reduce(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("sparse_reduce");

    for d in [4, 8] {
        let formula = product_sum_formula(d, 1);
        let spec = KernelSpec::new(formula, Iteration::Sparse, BindingOrder::LowToHigh);
        let kernel = compile::<Fr>(&spec);

        for log_n in [14, 18] {
            let num_pairs = 1usize << log_n;
            let n = num_pairs * 2;

            let keys: Vec<u64> = (0..num_pairs)
                .flat_map(|k| [k as u64 * 2, k as u64 * 2 + 1])
                .collect();

            let mut bufs: Vec<Buf<CpuBackend, Fr>> = (0..d)
                .map(|i| DeviceBuffer::Field(random_field_vec(n, 600 + i as u64)))
                .collect();
            bufs.push(DeviceBuffer::U64(keys));
            let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();

            group.throughput(Throughput::Elements(num_pairs as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| black_box(backend.reduce(&kernel, &buf_refs, &[])));
                },
            );
        }
    }

    group.finish();
}

fn bench_bind(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let mut group = c.benchmark_group("bind");

    for log_n in [14, 18] {
        let n = 1usize << log_n;
        let scalar = Fr::random(&mut rng);

        // Dense bind (LowToHigh)
        {
            let formula = product_sum_formula(4, 1);
            let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
            let kernel = compile::<Fr>(&spec);

            let raw_data: Vec<Vec<Fr>> = (0..4)
                .map(|i| random_field_vec(n, 800 + i))
                .collect();

            group.throughput(Throughput::Elements((n / 2) as u64));
            group.bench_with_input(
                BenchmarkId::new("dense_l2h", format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter_batched(
                        || {
                            raw_data
                                .iter()
                                .map(|v| DeviceBuffer::Field(v.clone()))
                                .collect::<Vec<Buf<CpuBackend, Fr>>>()
                        },
                        |mut bufs| {
                            backend.bind(&kernel, &mut bufs, scalar);
                            black_box(bufs);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }

        // Sparse bind
        {
            let formula = product_sum_formula(4, 1);
            let spec = KernelSpec::new(formula, Iteration::Sparse, BindingOrder::LowToHigh);
            let kernel = compile::<Fr>(&spec);

            let num_pairs = n / 2;
            let keys: Vec<u64> = (0..num_pairs)
                .flat_map(|k| [k as u64 * 2, k as u64 * 2 + 1])
                .collect();

            let raw_data: Vec<Vec<Fr>> = (0..4)
                .map(|i| random_field_vec(n, 900 + i))
                .collect();

            group.throughput(Throughput::Elements(num_pairs as u64));
            group.bench_with_input(
                BenchmarkId::new("sparse", format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter_batched(
                        || {
                            let mut bufs: Vec<Buf<CpuBackend, Fr>> = raw_data
                                .iter()
                                .map(|v| DeviceBuffer::Field(v.clone()))
                                .collect();
                            bufs.push(DeviceBuffer::U64(keys.clone()));
                            bufs
                        },
                        |mut bufs| {
                            backend.bind(&kernel, &mut bufs, scalar);
                            black_box(bufs);
                        },
                        criterion::BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_product_sum_kernels,
    bench_custom_kernel,
    bench_dense_reduce,
    bench_sparse_reduce,
    bench_bind
);
criterion_main!(benches);
