#![expect(unused_results)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use jolt_field::{Field, FpExt4, Fr, JoltField, Prime128Offset6A23801F, Prime32Offset99};
use jolt_poly::{Budget, EqPolynomial, PointRef, Polynomial, TensorEqTable, UnivariatePoly};
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
                        p.bind(std::hint::black_box(scalar));
                        p
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_bind_low_to_high_in_place(c: &mut Criterion) {
    bench_inplace_field::<Fr>(c, "Fr");
    bench_inplace_field::<Prime128Offset6A23801F>(c, "Prime128Offset6A23801F");
}

fn bench_inplace_field<F: JoltField>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("Polynomial::bind_low_to_high_in_place/{field}"));
    for num_vars in [14, 20, 24] {
        let mut rng = ChaCha20Rng::seed_from_u64(num_vars as u64);
        let poly = Polynomial::<F>::random(num_vars, &mut rng);
        let scalar = F::random(&mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter_batched(
                    || poly.clone(),
                    |mut p| {
                        p.bind_low_to_high_in_place(std::hint::black_box(scalar));
                        p
                    },
                    BatchSize::LargeInput,
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
                bench.iter(|| std::hint::black_box(&eq).evaluations());
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
        bench.iter(|| std::hint::black_box(&poly).evaluate(std::hint::black_box(&point)));
    });
}

fn bench_eq_evals(c: &mut Criterion) {
    let mut group = c.benchmark_group("EqPolynomial::evals");
    for num_vars in [17, 19, 20, 22] {
        let mut rng = ChaCha20Rng::seed_from_u64(300 + num_vars as u64);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter(|| {
                    EqPolynomial::<Fr>::evals(std::hint::black_box(&r), std::hint::black_box(None))
                });
            },
        );
    }
    group.finish();
}

fn bench_eq_orders(c: &mut Criterion) {
    bench_eq_field::<Fr>(c, "Fr");
    bench_eq_field::<Prime128Offset6A23801F>(c, "Prime128Offset6A23801F");
    bench_eq_field::<FpExt4<Prime32Offset99>>(c, "FpExt4");
}

fn bench_eq_field<F: Field>(c: &mut Criterion, field: &str) {
    let mut group = c.benchmark_group(format!("eq_orders/{field}"));
    for n in [12, 14, 16, 17, 20, 22] {
        let mut rng = ChaCha20Rng::seed_from_u64(300 + n as u64);
        let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        group.bench_function(BenchmarkId::new("be", n), |bench| {
            bench.iter(|| EqPolynomial::<F>::evals(std::hint::black_box(&r), None));
        });
        let le = PointRef::low_to_high(&r);
        group.bench_function(BenchmarkId::new("le", n), |bench| {
            bench.iter(|| EqPolynomial::<F>::evals(std::hint::black_box(&le), None));
        });
    }
    for n in [10, 14] {
        let mut rng = ChaCha20Rng::seed_from_u64(600 + n as u64);
        let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
        group.bench_function(BenchmarkId::new("cached_msb", n), |bench| {
            bench.iter(|| EqPolynomial::<F>::evals_cached_msb(std::hint::black_box(&r), None));
        });
        group.bench_function(BenchmarkId::new("cached_lsb", n), |bench| {
            bench.iter(|| EqPolynomial::<F>::evals_cached_lsb(std::hint::black_box(&r), None));
        });
        group.bench_function(BenchmarkId::new("tensor", n), |bench| {
            bench.iter(|| {
                TensorEqTable::<F>::new_with_scaling_and_split(
                    std::hint::black_box(&r),
                    F::from_u64(1),
                    n / 2,
                )
            });
        });
    }
    group.finish();
}

// Each case compares with `eq_orders/Fr/le` at the same size: the points share
// seeds and coordinate order.
fn bench_eq_builders(c: &mut Criterion) {
    let le_point = |n: usize| -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(300 + n as u64);
        (0..n).map(|_| Fr::random(&mut rng)).collect()
    };
    let budget = Budget {
        max_bytes: usize::MAX,
    };
    let mut group = c.benchmark_group("eq_builders/Fr");
    for n in [14, 17, 20] {
        let r = le_point(n);
        let point = PointRef::low_to_high(&r);
        group.bench_function(BenchmarkId::new("evals_budget", n), |bench| {
            bench.iter(|| EqPolynomial::<Fr>::evals_in(std::hint::black_box(&point), None, budget));
        });
    }
    for n in [16, 20] {
        let r = le_point(n);
        let point = PointRef::low_to_high(&r);
        group.bench_function(BenchmarkId::new("mapped", n), |bench| {
            bench.iter(|| {
                EqPolynomial::<Fr>::evals_mapped(std::hint::black_box(&point), None, |x| x * x)
            });
        });
    }
    let r = le_point(20);
    let point = PointRef::low_to_high(&r);
    group.bench_function(BenchmarkId::new("prefix_third", 20), |bench| {
        bench.iter(|| {
            EqPolynomial::<Fr>::evals_prefix(std::hint::black_box(&point), None, (1 << 20) / 3)
        });
    });
    group.finish();
}

fn bench_univariate_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("UnivariatePoly::from_evals");
    for num_evals in [3, 4, 8, 16] {
        let mut rng = ChaCha20Rng::seed_from_u64(400 + num_evals as u64);
        let evals: Vec<Fr> = (0..num_evals).map(|_| Fr::random(&mut rng)).collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(num_evals),
            &evals,
            |bench, evals| {
                bench.iter(|| UnivariatePoly::from_evals(std::hint::black_box(evals)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bind,
    bench_bind_low_to_high_in_place,
    bench_eq_evaluations,
    bench_evaluate,
    bench_eq_evals,
    bench_eq_orders,
    bench_eq_builders,
    bench_univariate_interpolation,
);
criterion_main!(benches);
