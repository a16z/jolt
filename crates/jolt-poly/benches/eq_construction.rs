#![expect(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, FpExt4, Fr, JoltField, Prime128Offset275, Prime32Offset99};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, PointRef, TensorEqTable};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn square<F: Field>(value: F) -> F {
    value * value
}

fn bench_eq_construction(c: &mut Criterion) {
    bench_eq_construction_field::<Fr>(c, "Fr");
    bench_eq_construction_field::<Prime128Offset275>(c, "Prime128Offset275");
    bench_eq_construction_field::<FpExt4<Prime32Offset99>>(c, "FpExt4");
    bench_eq_gruen_construction_field::<Fr>(c, "Fr");
    bench_eq_gruen_construction_field::<Prime128Offset275>(c, "Prime128Offset275");
}

fn bench_eq_construction_field<F: Field>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("eq_construction/{field}"));
    for n in [14, 16, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(0xe12 + n as u64);
        let coords: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let point = PointRef::low_to_high(&coords);
        if n != 14 {
            group.bench_function(BenchmarkId::new("evals_be", n), |bench| {
                bench.iter(|| EqPolynomial::<F>::evals(std::hint::black_box(&coords), None));
            });
            group.bench_function(BenchmarkId::new("evals_le", n), |bench| {
                bench.iter(|| EqPolynomial::<F>::evals(std::hint::black_box(&point), None));
            });
        }
        if n == 14 {
            group.bench_function(BenchmarkId::new("cached_msb", n), |bench| {
                bench.iter(|| {
                    EqPolynomial::<F>::evals_cached_msb(std::hint::black_box(&point), None)
                });
            });
            group.bench_function(BenchmarkId::new("cached_lsb", n), |bench| {
                bench.iter(|| {
                    EqPolynomial::<F>::evals_cached_lsb(std::hint::black_box(&point), None)
                });
            });
        }
        if n != 16 {
            group.bench_function(BenchmarkId::new("tensor", n), |bench| {
                bench.iter(|| {
                    TensorEqTable::<F>::new_with_scaling_and_split(
                        std::hint::black_box(&point),
                        F::from_u64(3),
                        n / 2,
                    )
                });
            });
        }
    }
    group.finish();
}

fn bench_eq_mapped(c: &mut Criterion) {
    bench_eq_mapped_field::<Fr>(c, "Fr");
    bench_eq_mapped_field::<Prime128Offset275>(c, "Prime128Offset275");
    bench_eq_mapped_field::<FpExt4<Prime32Offset99>>(c, "FpExt4");
}

fn bench_eq_mapped_field<F: Field>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("eq_construction/{field}"));
    for n in [16, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(0xe12 + n as u64);
        let coords: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let point = PointRef::low_to_high(&coords);
        group.bench_function(BenchmarkId::new("mapped", n), |bench| {
            bench.iter(|| {
                EqPolynomial::<F>::evals_mapped(std::hint::black_box(&point), None, square::<F>)
            });
        });
    }
    group.finish();
}

fn bench_eq_gruen_construction_field<F: JoltField>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("eq_construction/{field}"));
    for n in [16, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(0xe12 + n as u64);
        let coords: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        let point = PointRef::low_to_high(&coords);
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            group.bench_function(BenchmarkId::new(format!("gruen_{order:?}"), n), |bench| {
                bench.iter(|| {
                    GruenSplitEqPolynomial::<F>::new_with_scaling(
                        std::hint::black_box(&point),
                        order,
                        Some(F::from_u64(3)),
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_eq_construction, bench_eq_mapped);
criterion_main!(benches);
