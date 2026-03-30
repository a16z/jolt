#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_compiler::{Factor, Formula, ProductTerm};
use jolt_cpu::compile;
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

/// Helper: build a pure product-sum formula with `p` groups of `d` consecutive inputs.
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
            let kernel = compile::<Fr>(&formula);

            // Throughput = total input field elements processed per eval
            group.throughput(Throughput::Elements(total_inputs as u64));

            let num_evals = formula.degree();
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
    let kernel = compile::<Fr>(&formula);
    let (lo, hi) = random_vecs(1, 999);

    let num_evals = formula.degree();
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
    let kernel = compile::<Fr>(&formula);
    let (lo, hi) = random_vecs(4, 1000);

    let num_evals = formula.degree();
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

criterion_group!(benches, bench_product_sum_kernels, bench_custom_kernel);
criterion_main!(benches);
