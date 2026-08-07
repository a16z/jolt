#![expect(unused_results)]

use std::hint::black_box;

use akita_config::proof_optimized::fp128::Field as AkitaField;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::signed::S256;
use jolt_field::{
    AdditiveAccumulator, AkitaAccumulator, AkitaSignedProductAccumulator,
    AkitaSmallScalarAccumulator, Fr, FrSignedProductAccumulator, FrSmallScalarAccumulator,
    NaiveAccumulator, NaiveSignedProductAccumulator, NaiveSignedScalarAccumulator, RandomSampling,
    RingAccumulator, SignedProductAccumulator, SignedScalarAccumulator, WideAccumulator,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

const TERMS: usize = 1 << 10;

fn signed_products(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(10);
    let fp128_values = (0..TERMS)
        .map(|_| AkitaField::random(&mut rng))
        .collect::<Vec<_>>();
    let bn254_values = (0..TERMS).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
    let scalars = (0..TERMS)
        .map(|index| {
            S256::new(
                [rng.next_u64(), rng.next_u64(), rng.next_u64(), 0],
                index % 2 == 0,
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("signed_product_accumulator");
    group.throughput(Throughput::Elements(TERMS as u64));
    group.bench_function(BenchmarkId::new("fp128_naive", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = NaiveSignedProductAccumulator::<AkitaField>::default();
            for (&value, scalar) in fp128_values.iter().zip(&scalars) {
                accumulator.fmadd_s256(black_box(value), black_box(scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("fp128_solinas", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = AkitaSignedProductAccumulator::default();
            for (&value, scalar) in fp128_values.iter().zip(&scalars) {
                accumulator.fmadd_s256(black_box(value), black_box(scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("bn254_wide", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = FrSignedProductAccumulator::default();
            for (&value, scalar) in bn254_values.iter().zip(&scalars) {
                accumulator.fmadd_s256(black_box(value), black_box(scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.finish();
}

fn signed_u64_products(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let values = (0..TERMS)
        .map(|_| AkitaField::random(&mut rng))
        .collect::<Vec<_>>();
    let scalars = (0..TERMS)
        .map(|index| (rng.next_u64(), index % 2 == 0))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("signed_u64_product_accumulator");
    group.throughput(Throughput::Elements(TERMS as u64));
    group.bench_function(BenchmarkId::new("fp128_wide", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = AkitaSignedProductAccumulator::default();
            for (&value, &(magnitude, is_positive)) in values.iter().zip(&scalars) {
                let scalar = S256::new([magnitude, 0, 0, 0], is_positive);
                accumulator.fmadd_s256(black_box(value), black_box(&scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("fp128_one_limb", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = AkitaSignedProductAccumulator::default();
            for (&value, &(magnitude, is_positive)) in values.iter().zip(&scalars) {
                accumulator.fmadd_signed_u64(
                    black_box(value),
                    black_box(magnitude),
                    black_box(is_positive),
                );
            }
            black_box(accumulator.reduce())
        });
    });
    group.finish();
}

fn small_scalars(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let fp128_values = (0..TERMS)
        .map(|_| AkitaField::random(&mut rng))
        .collect::<Vec<_>>();
    let bn254_values = (0..TERMS).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
    let scalars = (0..TERMS)
        .map(|_| rng.next_u64() as i64)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("small_scalar_accumulator");
    group.throughput(Throughput::Elements(TERMS as u64));
    group.bench_function(BenchmarkId::new("fp128_naive", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = NaiveSignedScalarAccumulator::<AkitaField>::default();
            for (&value, &scalar) in fp128_values.iter().zip(&scalars) {
                accumulator.fmadd_i64(black_box(value), black_box(scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("fp128_solinas", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = AkitaSmallScalarAccumulator::default();
            for (&value, &scalar) in fp128_values.iter().zip(&scalars) {
                accumulator.fmadd_i64(black_box(value), black_box(scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("bn254_wide", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = FrSmallScalarAccumulator::default();
            for (&value, &scalar) in bn254_values.iter().zip(&scalars) {
                accumulator.fmadd_i64(black_box(value), black_box(scalar));
            }
            black_box(accumulator.reduce())
        });
    });
    group.finish();
}

fn field_products(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(12);
    let fp128 = (0..TERMS)
        .map(|_| (AkitaField::random(&mut rng), AkitaField::random(&mut rng)))
        .collect::<Vec<_>>();
    let bn254 = (0..TERMS)
        .map(|_| (Fr::random(&mut rng), Fr::random(&mut rng)))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("field_product_accumulator");
    group.throughput(Throughput::Elements(TERMS as u64));
    group.bench_function(BenchmarkId::new("fp128_naive", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = NaiveAccumulator::<AkitaField>::default();
            for &(a, b) in &fp128 {
                accumulator.fmadd(black_box(a), black_box(b));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("fp128_solinas", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = AkitaAccumulator::default();
            for &(a, b) in &fp128 {
                accumulator.fmadd(black_box(a), black_box(b));
            }
            black_box(accumulator.reduce())
        });
    });
    group.bench_function(BenchmarkId::new("bn254_wide", TERMS), |bench| {
        bench.iter(|| {
            let mut accumulator = WideAccumulator::default();
            for &(a, b) in &bn254 {
                accumulator.fmadd(black_box(a), black_box(b));
            }
            black_box(accumulator.reduce())
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    signed_products,
    signed_u64_products,
    small_scalars,
    field_products
);
criterion_main!(benches);
