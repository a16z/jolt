#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{BindingOrder, Factor, Formula, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_field_vec(n: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

/// Build a product-sum formula with `d` factors per group and `p` groups.
fn product_sum_formula(d: usize, p: usize) -> Formula {
    let terms: Vec<_> = (0..p)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

/// Build a KernelSpec for a product-sum of degree `d` with `p` groups.
fn product_sum_spec(d: usize, p: usize) -> KernelSpec {
    KernelSpec::new(
        product_sum_formula(d, p),
        Iteration::Dense,
        BindingOrder::LowToHigh,
    )
}

/// Wrap field vectors as DeviceBuffer refs for `reduce`.
fn wrap_field_bufs(vecs: &[Vec<Fr>]) -> Vec<Buf<CpuBackend, Fr>> {
    vecs.iter()
        .map(|v| DeviceBuffer::Field(v.clone()))
        .collect()
}

fn bench_interpolate_inplace(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("interpolate_inplace");

    for log_n in [16, 18, 20] {
        let n = 1usize << log_n;
        let half = (n / 2) as u64;
        let data = random_field_vec(n, 42);
        let scalar = Fr::from_u64(0x1234_5678);

        group.throughput(Throughput::Elements(half));

        group.bench_with_input(
            BenchmarkId::new("low_to_high", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut buf = data.clone();
                    backend.interpolate_inplace(&mut buf, scalar, BindingOrder::LowToHigh);
                    black_box(&buf);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("high_to_low", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut buf = data.clone();
                    backend.interpolate_inplace(&mut buf, scalar, BindingOrder::HighToLow);
                    black_box(&buf);
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

fn bench_reduce(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("reduce");

    for d in [4, 8, 16] {
        let spec = product_sum_spec(d, 1);
        let kernel = backend.compile::<Fr>(&spec);

        for log_n in [16, 18, 20] {
            let n = 1usize << log_n;
            let half = n / 2;

            let inputs: Vec<Vec<Fr>> = (0..d)
                .map(|i| random_field_vec(n, 100 + i as u64))
                .collect();
            let dev_bufs = wrap_field_bufs(&inputs);
            let refs: Vec<&Buf<CpuBackend, Fr>> = dev_bufs.iter().collect();

            group.throughput(Throughput::Elements(half as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| {
                        black_box(backend.reduce(&kernel, &refs, &[]));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_bind(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("bind");

    for d in [4, 8] {
        let spec = product_sum_spec(d, 1);
        let kernel = backend.compile::<Fr>(&spec);

        for log_n in [16, 18, 20] {
            let n = 1usize << log_n;
            let half = (n / 2) as u64;

            let inputs: Vec<Vec<Fr>> = (0..d)
                .map(|i| random_field_vec(n, 100 + i as u64))
                .collect();
            let scalar = Fr::from_u64(0xABCD);

            group.throughput(Throughput::Elements(half * d as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("D={d}"), format!("2^{log_n}")),
                &n,
                |b, _| {
                    b.iter(|| {
                        let mut dev_bufs = wrap_field_bufs(&inputs);
                        backend.bind(&kernel, &mut dev_bufs, scalar);
                        black_box(&dev_bufs);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_tensor_reduce(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("tensor_reduce");

    let d = 4;
    let formula = product_sum_formula(d, 1);

    // Test various outer x inner splits at total ~2^18 pairs
    for (outer_log, inner_log) in [(5, 13), (9, 9), (13, 5)] {
        let outer_len = 1usize << outer_log;
        let inner_len = 1usize << inner_log;
        let total_pairs = outer_len * inner_len;
        let n = total_pairs * 2;

        let inputs: Vec<Vec<Fr>> = (0..d)
            .map(|i| random_field_vec(n, 100 + i as u64))
            .collect();
        let outer_w = random_field_vec(outer_len, 200);
        let inner_w = random_field_vec(inner_len, 201);

        // Flat weights for comparison (dense reduce with materialized eq)
        let mut flat_w = Vec::with_capacity(total_pairs);
        for &o in &outer_w {
            for &i in &inner_w {
                flat_w.push(o * i);
            }
        }

        group.throughput(Throughput::Elements(total_pairs as u64));

        let label = format!("{outer_log}+{inner_log}");

        // Dense kernel with materialized eq as an extra input column
        let dense_formula = {
            // eq * (product of d inputs) — eq is input(d), value inputs are 0..d
            let mut factors: Vec<Factor> = (0..d).map(|j| Factor::Input(j as u32)).collect();
            factors.push(Factor::Input(d as u32));
            Formula::from_terms(vec![ProductTerm {
                coefficient: 1,
                factors,
            }])
        };
        let dense_spec = KernelSpec::new(dense_formula, Iteration::Dense, BindingOrder::LowToHigh);
        let dense_kernel = backend.compile::<Fr>(&dense_spec);

        // Build flat eq buffer: interleave pairs for LowToHigh binding
        let mut flat_eq_buf = vec![Fr::from_u64(0); n];
        for i in 0..total_pairs {
            flat_eq_buf[2 * i] = flat_w[i];
            flat_eq_buf[2 * i + 1] = flat_w[i];
        }

        let mut flat_dev_bufs: Vec<Buf<CpuBackend, Fr>> = inputs
            .iter()
            .map(|v| DeviceBuffer::Field(v.clone()))
            .collect();
        flat_dev_bufs.push(DeviceBuffer::Field(flat_eq_buf));
        let flat_refs: Vec<&Buf<CpuBackend, Fr>> = flat_dev_bufs.iter().collect();

        group.bench_with_input(BenchmarkId::new("flat", &label), &total_pairs, |b, _| {
            b.iter(|| {
                black_box(backend.reduce(&dense_kernel, &flat_refs, &[]));
            });
        });

        // Tensor kernel
        let tensor_spec = KernelSpec::new(
            formula.clone(),
            Iteration::DenseTensor,
            BindingOrder::LowToHigh,
        );
        let tensor_kernel = backend.compile::<Fr>(&tensor_spec);

        let mut tensor_dev_bufs: Vec<Buf<CpuBackend, Fr>> = inputs
            .iter()
            .map(|v| DeviceBuffer::Field(v.clone()))
            .collect();
        tensor_dev_bufs.push(DeviceBuffer::Field(outer_w.clone()));
        tensor_dev_bufs.push(DeviceBuffer::Field(inner_w.clone()));
        let tensor_refs: Vec<&Buf<CpuBackend, Fr>> = tensor_dev_bufs.iter().collect();

        group.bench_with_input(BenchmarkId::new("tensor", &label), &total_pairs, |b, _| {
            b.iter(|| {
                black_box(backend.reduce(&tensor_kernel, &tensor_refs, &[]));
            });
        });
    }

    group.finish();
}

fn bench_eq_table(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("eq_table");

    for n_vars in [16, 20, 24] {
        let table_size = 1u64 << n_vars;
        let point = random_field_vec(n_vars, 42);

        group.throughput(Throughput::Elements(table_size));

        group.bench_with_input(
            BenchmarkId::new("compute", format!("n={n_vars}")),
            &n_vars,
            |b, _| {
                b.iter(|| {
                    black_box(backend.eq_table(&point));
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

fn bench_compile(c: &mut Criterion) {
    let backend = CpuBackend;
    let mut group = c.benchmark_group("compile");

    for d in [4, 8, 16, 32] {
        let spec = product_sum_spec(d, 1);

        group.bench_with_input(
            BenchmarkId::new("product_sum", format!("D={d}")),
            &d,
            |b, _| {
                b.iter(|| {
                    black_box(backend.compile::<Fr>(&spec));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_interpolate_inplace,
    bench_reduce,
    bench_bind,
    bench_eq_table,
    bench_tensor_reduce,
    bench_compile,
);
criterion_main!(benches);
