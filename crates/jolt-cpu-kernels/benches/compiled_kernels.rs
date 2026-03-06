#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_cpu_kernels::compile;
use jolt_field::{Field, Fr};
use jolt_ir::{KernelDescriptor, KernelShape};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn random_vecs(n: usize, seed: u64) -> (Vec<Fr>, Vec<Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let lo: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let hi: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    (lo, hi)
}

fn bench_product_sum_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_sum_kernel_eval");

    for d in [4, 8, 16] {
        for num_products in [1, 4] {
            let total_inputs = d * num_products;
            let (lo, hi) = random_vecs(total_inputs, d as u64 * 100 + num_products as u64);

            let desc = KernelDescriptor {
                shape: KernelShape::ProductSum {
                    num_inputs_per_product: d,
                    num_products,
                },
                degree: d,
                tensor_split: None,
            };
            let kernel = compile::<Fr>(&desc);

            group.bench_with_input(
                BenchmarkId::new(format!("D={d}/P={num_products}"), "specialized"),
                &d,
                |b, _| {
                    b.iter(|| {
                        black_box(kernel.evaluate(&lo, &hi, d));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_custom_kernel(c: &mut Criterion) {
    use jolt_ir::ExprBuilder;

    let mut group = c.benchmark_group("custom_kernel_eval");

    // Booleanity: o0^2 - o0
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(h * h - h),
            num_inputs: 1,
        },
        degree: 2,
        tensor_split: None,
    };
    let kernel = compile::<Fr>(&desc);
    let (lo, hi) = random_vecs(1, 999);

    group.bench_function("booleanity", |bench| {
        bench.iter(|| {
            black_box(kernel.evaluate(&lo, &hi, 2));
        });
    });

    // Product: o0 * o1 * o2 * o3
    let b = ExprBuilder::new();
    let o0 = b.opening(0);
    let o1 = b.opening(1);
    let o2 = b.opening(2);
    let o3 = b.opening(3);
    let desc = KernelDescriptor {
        shape: KernelShape::Custom {
            expr: b.build(o0 * o1 * o2 * o3),
            num_inputs: 4,
        },
        degree: 4,
        tensor_split: None,
    };
    let kernel = compile::<Fr>(&desc);
    let (lo, hi) = random_vecs(4, 1000);

    group.bench_function("product_4_via_custom", |bench| {
        bench.iter(|| {
            black_box(kernel.evaluate(&lo, &hi, 4));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_product_sum_kernels, bench_custom_kernel);
criterion_main!(benches);
