use ark_bn254::Fr;
use ark_ff::Zero;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use jolt_core::field::challenge::{Mont254BitChallenge, MontU128Challenge};
use jolt_core::field::JoltField;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bench_add(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(123);

    c.bench_function(
        "(x - a) * b (100x): where x: Fr, a,b: Mont254BitChallenge Challenge",
        |b| {
            b.iter_batched(
                || {
                    let x = <Fr as JoltField>::random(&mut rng);
                    let ab: Vec<_> = (0..100)
                        .map(|_| {
                            (
                                Mont254BitChallenge::random(&mut rng),
                                Mont254BitChallenge::random(&mut rng),
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

    c.bench_function(
        "(x - a) * b (100x): where x: Fr, a,b: MontU128 Challenge",
        |b| {
            b.iter_batched(
                || {
                    let x = <Fr as JoltField>::random(&mut rng);
                    let ab: Vec<_> = (0..100)
                        .map(|_| {
                            (
                                MontU128Challenge::random(&mut rng),
                                MontU128Challenge::random(&mut rng),
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
    c.bench_function("Fr * Mont254BitChallenge (100x)", |b| {
        b.iter_batched(
            || {
                let a = <Fr as JoltField>::random(&mut rng);
                let bs: Vec<_> = (0..100)
                    .map(|_| Mont254BitChallenge::random(&mut rng))
                    .collect();
                (a, bs)
            },
            |(mut acc, bs)| {
                for b in bs {
                    acc = acc * b;
                }
                acc
            },
            BatchSize::SmallInput,
        )
    });

    // Fr * u128 via from_u128_mont
    c.bench_function(
        "Fr * <u128 as Fr> (via from_u128_mont) (100x): (No Speedups expected)",
        |b| {
            b.iter_batched(
                || {
                    let a = <Fr as JoltField>::random(&mut rng);
                    let xs: Vec<Fr> = (0..100)
                        .map(|_| MontU128Challenge::random(&mut rng).into())
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
        },
    );

    // Fr.mul_u128_mont_form
    c.bench_function("Fr.mul_u128_mont_form (100x)", |b| {
        b.iter_batched(
            || {
                let a = <Fr as JoltField>::random(&mut rng);
                let xs: Vec<_> = (0..100)
                    .map(|_| MontU128Challenge::random(&mut rng))
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

// Polynomial binding benchmarks use F::Challenge which is controlled by feature flags.
// Default: MontU128Challenge (optimized)
// With --features challenge-254-bit: Mont254BitChallenge (baseline)
fn bench_poly_binding(c: &mut Criterion) {
    let mut group = c.benchmark_group("poly_binding");

    // Only test 2^20 for stable measurements
    let num_vars = 20;
    let poly_size = 1 << num_vars;

    // Benchmark HighToLow binding
    group.bench_with_input(
        BenchmarkId::new("high_to_low", format!("2^{num_vars}")),
        &poly_size,
        |b, &size| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter_batched(
                || {
                    let coeffs: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
                    let poly = DensePolynomial::new(coeffs);
                    let challenge = <Fr as JoltField>::Challenge::random(&mut rng);
                    (poly, challenge)
                },
                |(mut poly, challenge)| {
                    poly.bind_parallel(challenge, BindingOrder::HighToLow);
                    poly
                },
                BatchSize::LargeInput,
            )
        },
    );

    // Benchmark LowToHigh binding
    group.bench_with_input(
        BenchmarkId::new("low_to_high", format!("2^{num_vars}")),
        &poly_size,
        |b, &size| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter_batched(
                || {
                    let coeffs: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
                    let poly = DensePolynomial::new(coeffs);
                    let challenge = <Fr as JoltField>::Challenge::random(&mut rng);
                    (poly, challenge)
                },
                |(mut poly, challenge)| {
                    poly.bind_parallel(challenge, BindingOrder::LowToHigh);
                    poly
                },
                BatchSize::LargeInput,
            )
        },
    );

    group.finish();
}

criterion_group!(benches, bench_mul, bench_add, bench_poly_binding);
criterion_main!(benches);
