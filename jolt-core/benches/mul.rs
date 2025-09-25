use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField, Zero};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jolt_core::field::challenge::MontU128Challenge;
use jolt_core::field::JoltField;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn bench_add(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(123);

    // Fr case: (x - a) * b, where a, b ∈ Fr
    c.bench_function("Fr: (x - a) * b (100x)", |b| {
        b.iter_batched(
            || {
                let x = Fr::random(&mut rng);
                let ab: Vec<_> = (0..100)
                    .map(|_| (Fr::random(&mut rng), Fr::random(&mut rng)))
                    .collect();
                (x, ab)
            },
            |(x, ab)| {
                let mut acc = Fr::zero();
                for (a, b) in ab {
                    acc += (x - a) * b;
                }
                acc
            },
            BatchSize::SmallInput,
        )
    });

    // MontU128 case: (x - from_u128_mont(a)) * mul_u128_mont_form(b), where a, b ∈ MontU128
    c.bench_function(
        "Fr: (x - from_u128_mont(a)) * mul_u128_mont_form(b) (100x)",
        |b| {
            b.iter_batched(
                || {
                    let x = Fr::random(&mut rng);
                    let ab: Vec<_> = (0..100)
                        .map(|_| {
                            (
                                MontU128Challenge::from(rng.gen::<u128>()),
                                MontU128Challenge::from(rng.gen::<u128>()),
                            )
                        })
                        .collect();
                    (x, ab)
                },
                |(x, ab)| {
                    let mut acc = Fr::zero();
                    for (a, b) in ab {
                        acc += (x - a) * b;
                    }
                    acc
                },
                BatchSize::SmallInput,
            )
        },
    );
}

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
                let xs: Vec<Fr> = (0..100)
                    .map(|_| {
                        let tmp = BigInt(MontU128Challenge::from(rng.gen::<u128>()).value());
                        Fr::from_bigint_unchecked(tmp).unwrap()
                    })
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
                    .map(|_| MontU128Challenge::from(rng.gen::<u128>()))
                    .collect();
                (a, xs)
            },
            |(mut acc, xs)| {
                for x in xs {
                    acc = acc * x;
                }
                acc
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_mul, bench_add);
criterion_main!(benches);
