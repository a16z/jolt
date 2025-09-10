use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_core::field::{JoltField, MontU128};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn bench_mul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(123);

    c.bench_function("Fr * Fr", |b| {
        let a = Fr::random(&mut rng);
        let b_fr = Fr::random(&mut rng);
        b.iter(|| {
            let _ = a * b_fr;
        })
    });

    c.bench_function("Fr * u128 (via from_u128_mont) - slowest", |b| {
        let a = Fr::random(&mut rng);
        let x = MontU128::from(rng.gen::<u128>());
        let b_fr = Fr::from_u128_mont(x);
        b.iter(|| {
            let _ = a * b_fr;
        })
    });

    c.bench_function("Fr.mul_u128_mont_form", |b| {
        let a = Fr::random(&mut rng);
        let x = MontU128::from(rng.gen::<u128>());
        b.iter(|| {
            let _ = a.mul_u128_mont_form(x);
        })
    });
}

criterion_group!(benches, bench_mul);
criterion_main!(benches);
