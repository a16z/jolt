#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, Polynomial};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("Polynomial::bind");
    for num_vars in [14, 18, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(num_vars as u64);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let scalar = Fr::random(&mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter_batched(
                    || poly.clone(),
                    |mut p| {
                        p.bind(black_box(scalar));
                        p
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_eq_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("EqPolynomial::evaluations");
    for num_vars in [14, 18, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(100 + num_vars as u64);
        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter(|| black_box(&eq).evaluations());
            },
        );
    }
    group.finish();
}

fn bench_evaluate(c: &mut Criterion) {
    let num_vars = 20;
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    c.bench_function("Polynomial::evaluate/20", |bench| {
        bench.iter(|| black_box(&poly).evaluate(black_box(&point)));
    });
}

criterion_group!(benches, bench_bind, bench_eq_evaluations, bench_evaluate);
criterion_main!(benches);
