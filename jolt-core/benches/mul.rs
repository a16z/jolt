use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jolt_core::field::{JoltField, MontU128};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn bench_mul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(123);

    // Fr * Fr
    c.bench_function("Fr * Fr (100x)", |b| {
        b.iter_batched(
            || {
                let a = Fr::random(&mut rng);
                let bs: Vec<_> = (0..100).map(|_| Fr::random(&mut rng)).collect();
                (a, bs)
            },
            |(mut acc, bs)| {
                for b in bs {
                    acc *= b;
                }
                acc
            },
            BatchSize::SmallInput,
        )
    });

    // Fr * u128 via from_u128_mont
    c.bench_function("Fr * u128 (via from_u128_mont) (100x)", |b| {
        b.iter_batched(
            || {
                let a = Fr::random(&mut rng);
                let xs: Vec<_> = (0..100)
                    .map(|_| Fr::from_u128_mont(MontU128::from(rng.gen::<u128>())))
                    .collect();
                (a, xs)
            },
            |(mut acc, xs)| {
                for x in xs {
                    acc *= x;
                }
                acc
            },
            BatchSize::SmallInput,
        )
    });

    // Fr.mul_u128_mont_form
    c.bench_function("Fr.mul_u128_mont_form (100x)", |b| {
        b.iter_batched(
            || {
                let a = Fr::random(&mut rng);
                let xs: Vec<_> = (0..100)
                    .map(|_| MontU128::from(rng.gen::<u128>()))
                    .collect();
                (a, xs)
            },
            |(mut acc, xs)| {
                for x in xs {
                    acc = acc.mul_u128_mont_form(x);
                }
                acc
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_mul);
criterion_main!(benches);
