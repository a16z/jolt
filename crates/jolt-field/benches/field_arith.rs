#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jolt_field::challenge::MontU128Challenge;
use jolt_field::{Field, Fr, OptimizedMul};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn bench_field_mul(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let a: Fr = Field::random(&mut rng);
    let b: Fr = Field::random(&mut rng);

    c.bench_function("Fr * Fr", |bench| {
        bench.iter(|| black_box(a) * black_box(b));
    });
}

fn bench_mul_u64(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let a: Fr = Field::random(&mut rng);
    let n = 0xDEAD_BEEF_CAFE_BABEu64;

    c.bench_function("Fr::mul_u64", |bench| {
        bench.iter(|| <Fr as Field>::mul_u64(black_box(&a), black_box(n)));
    });
}

fn bench_mul_u128(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    let a: Fr = Field::random(&mut rng);
    let n = 0xDEAD_BEEF_CAFE_BABE_1234_5678_9ABC_DEF0u128;

    c.bench_function("Fr::mul_u128", |bench| {
        bench.iter(|| <Fr as Field>::mul_u128(black_box(&a), black_box(n)));
    });
}

fn bench_challenge_mul_fr(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let challenge = MontU128Challenge::<Fr>::random(&mut rng);
    let field: Fr = Field::random(&mut rng);

    c.bench_function("MontU128Challenge * Fr", |bench| {
        bench.iter(|| black_box(challenge) * black_box(field));
    });
}

fn bench_to_from_bytes(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    let a: Fr = Field::random(&mut rng);
    let bytes = a.to_bytes();

    c.bench_function("Fr::to_bytes", |bench| {
        bench.iter(|| black_box(a).to_bytes());
    });

    c.bench_function("Fr::from_bytes", |bench| {
        bench.iter(|| <Fr as Field>::from_bytes(black_box(&bytes)));
    });
}

fn bench_challenge_mul_01_optimized(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let challenge = MontU128Challenge::<Fr>::random(&mut rng);
    let field: Fr = Field::random(&mut rng);

    c.bench_function("MontU128Challenge::mul_01_optimized", |bench| {
        bench.iter(|| black_box(challenge).mul_01_optimized(black_box(field)));
    });
}

criterion_group!(
    benches,
    bench_field_mul,
    bench_mul_u64,
    bench_mul_u128,
    bench_challenge_mul_fr,
    bench_to_from_bytes,
    bench_challenge_mul_01_optimized,
);
criterion_main!(benches);
