use std::{hint::black_box, time::Duration};

use ark_bn254::Fr;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_prover_legacy::field::{akita::AkitaFp128, JoltField};
use num_traits::Zero;
use rand::{rngs::StdRng, SeedableRng};

const BOOLEANITY_COLUMNS: usize = 32;
const D4_PRODUCTS: usize = 4;

fn random_fields<F: JoltField>(len: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    std::iter::repeat_with(|| F::random(&mut rng))
        .take(len)
        .collect()
}

#[inline(never)]
fn dot_product_raw<F: JoltField>(left: &[F], right: &[F]) -> F {
    let mut acc = F::UnreducedProductAccum::zero();
    for (&a, &b) in left.iter().zip(right) {
        acc += a.mul_to_product_accum(b);
    }
    F::reduce_product_accum(acc)
}

#[inline(never)]
fn dot_product_eager<F: JoltField>(left: &[F], right: &[F]) -> F {
    left.iter()
        .zip(right)
        .fold(F::zero(), |acc, (&a, &b)| acc + a * b)
}

#[inline(never)]
fn dot_product_multi<F: JoltField>(left: &[F], right: &[F]) -> F {
    let mut acc = F::MultiProductAccum::zero();
    for (&a, &b) in left.iter().zip(right) {
        acc += a.mul_to_multi_product_accum(b);
    }
    F::reduce_multi_product_accum(acc)
}

struct BooleanityInput<F> {
    h0: Vec<F>,
    h1: Vec<F>,
    rho: Vec<F>,
    weights: Vec<F>,
}

fn booleanity_input<F: JoltField>(rows: usize, seed: u64) -> BooleanityInput<F> {
    BooleanityInput {
        h0: random_fields(rows * BOOLEANITY_COLUMNS, seed),
        h1: random_fields(rows * BOOLEANITY_COLUMNS, seed + 1),
        rho: random_fields(BOOLEANITY_COLUMNS, seed + 2),
        weights: random_fields(rows, seed + 3),
    }
}

#[inline(never)]
fn booleanity_raw<F: JoltField>(input: &BooleanityInput<F>) -> [F; 2] {
    let mut outer = [F::UnreducedProductAccum::zero(); 2];
    for (row, &weight) in input.weights.iter().enumerate() {
        let base = row * BOOLEANITY_COLUMNS;
        let mut inner = [F::UnreducedProductAccum::zero(); 2];
        for column in 0..BOOLEANITY_COLUMNS {
            let h0 = input.h0[base + column];
            let delta = input.h1[base + column] - h0;
            inner[0] += h0.mul_to_product_accum(h0 - input.rho[column]);
            inner[1] += delta.mul_to_product_accum(delta);
        }
        outer[0] += weight.mul_to_product_accum(F::reduce_product_accum(inner[0]));
        outer[1] += weight.mul_to_product_accum(F::reduce_product_accum(inner[1]));
    }
    outer.map(F::reduce_product_accum)
}

#[inline(never)]
fn booleanity_eager<F: JoltField>(input: &BooleanityInput<F>) -> [F; 2] {
    let mut outer = [F::zero(); 2];
    for (row, &weight) in input.weights.iter().enumerate() {
        let base = row * BOOLEANITY_COLUMNS;
        let mut inner = [F::zero(); 2];
        for column in 0..BOOLEANITY_COLUMNS {
            let h0 = input.h0[base + column];
            let delta = input.h1[base + column] - h0;
            inner[0] += h0 * (h0 - input.rho[column]);
            inner[1] += delta * delta;
        }
        outer[0] += weight * inner[0];
        outer[1] += weight * inner[1];
    }
    outer
}

#[inline(never)]
fn booleanity_multi<F: JoltField>(input: &BooleanityInput<F>) -> [F; 2] {
    let mut outer = [F::MultiProductAccum::zero(); 2];
    for (row, &weight) in input.weights.iter().enumerate() {
        let base = row * BOOLEANITY_COLUMNS;
        let mut inner = [F::MultiProductAccum::zero(); 2];
        for column in 0..BOOLEANITY_COLUMNS {
            let h0 = input.h0[base + column];
            let delta = input.h1[base + column] - h0;
            inner[0] += h0.mul_to_multi_product_accum(h0 - input.rho[column]);
            inner[1] += delta.mul_to_multi_product_accum(delta);
        }
        outer[0] += weight.mul_to_multi_product_accum(F::reduce_multi_product_accum(inner[0]));
        outer[1] += weight.mul_to_multi_product_accum(F::reduce_multi_product_accum(inner[1]));
    }
    outer.map(F::reduce_multi_product_accum)
}

struct D4Input<F> {
    pairs: Vec<(F, F)>,
    weights: Vec<F>,
}

fn d4_input<F: JoltField>(rows: usize, seed: u64) -> D4Input<F> {
    let values = random_fields::<F>(rows * D4_PRODUCTS * 8, seed);
    let pairs = values
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    D4Input {
        pairs,
        weights: random_fields(rows, seed + 1),
    }
}

#[inline(always)]
fn quadratic_product_evals<F: JoltField>(left: (F, F), right: (F, F)) -> [F; 4] {
    let left_inf = left.1 - left.0;
    let right_inf = right.1 - right.0;
    let at_1 = left.1 * right.1;
    let at_2 = (left.1 + left_inf) * (right.1 + right_inf);
    let at_inf = left_inf * right_inf;
    let at_3 = (at_2 + at_inf) + (at_2 + at_inf) - at_1;
    [at_1, at_2, at_3, at_inf]
}

#[inline(always)]
fn d4_halves<F: JoltField>(pairs: &[(F, F)]) -> ([F; 4], [F; 4]) {
    (
        quadratic_product_evals(pairs[0], pairs[1]),
        quadratic_product_evals(pairs[2], pairs[3]),
    )
}

#[inline(never)]
fn d4_raw<F: JoltField>(input: &D4Input<F>) -> [F; 4] {
    let mut outer = [F::UnreducedProductAccum::zero(); 4];
    for (row, &weight) in input.weights.iter().enumerate() {
        let row_base = row * D4_PRODUCTS * 4;
        let mut inner = [F::UnreducedProductAccum::zero(); 4];
        for product in 0..D4_PRODUCTS {
            let base = row_base + product * 4;
            let (left, right) = d4_halves(&input.pairs[base..base + 4]);
            for lane in 0..4 {
                inner[lane] += left[lane].mul_to_product_accum(right[lane]);
            }
        }
        for lane in 0..4 {
            outer[lane] += weight.mul_to_product_accum(F::reduce_product_accum(inner[lane]));
        }
    }
    outer.map(F::reduce_product_accum)
}

#[inline(never)]
fn d4_eager<F: JoltField>(input: &D4Input<F>) -> [F; 4] {
    let mut outer = [F::zero(); 4];
    for (row, &weight) in input.weights.iter().enumerate() {
        let row_base = row * D4_PRODUCTS * 4;
        let mut inner = [F::zero(); 4];
        for product in 0..D4_PRODUCTS {
            let base = row_base + product * 4;
            let (left, right) = d4_halves(&input.pairs[base..base + 4]);
            for lane in 0..4 {
                inner[lane] += left[lane] * right[lane];
            }
        }
        for lane in 0..4 {
            outer[lane] += weight * inner[lane];
        }
    }
    outer
}

#[inline(never)]
fn d4_multi<F: JoltField>(input: &D4Input<F>) -> [F; 4] {
    let mut outer = [F::MultiProductAccum::zero(); 4];
    for (row, &weight) in input.weights.iter().enumerate() {
        let row_base = row * D4_PRODUCTS * 4;
        let mut inner = [F::MultiProductAccum::zero(); 4];
        for product in 0..D4_PRODUCTS {
            let base = row_base + product * 4;
            let (left, right) = d4_halves(&input.pairs[base..base + 4]);
            for lane in 0..4 {
                inner[lane] += left[lane].mul_to_multi_product_accum(right[lane]);
            }
        }
        for lane in 0..4 {
            outer[lane] +=
                weight.mul_to_multi_product_accum(F::reduce_multi_product_accum(inner[lane]));
        }
    }
    outer.map(F::reduce_multi_product_accum)
}

fn bench_dot_products(c: &mut Criterion) {
    for (label, len) in [("cache", 1 << 12), ("stream", 1 << 20)] {
        let akita_left = random_fields::<AkitaFp128>(len, 1);
        let akita_right = random_fields::<AkitaFp128>(len, 2);
        let bn_left = random_fields::<Fr>(len, 1);
        let bn_right = random_fields::<Fr>(len, 2);

        let expected = dot_product_eager(&akita_left, &akita_right);
        assert_eq!(dot_product_raw(&akita_left, &akita_right), expected);
        assert_eq!(dot_product_multi(&akita_left, &akita_right), expected);

        let mut group = c.benchmark_group(format!("dot_product/{label}"));
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function("akita_raw", |b| {
            b.iter(|| black_box(dot_product_raw(&akita_left, &akita_right)))
        });
        group.bench_function("akita_eager", |b| {
            b.iter(|| black_box(dot_product_eager(&akita_left, &akita_right)))
        });
        group.bench_function("akita_solinas", |b| {
            b.iter(|| black_box(dot_product_multi(&akita_left, &akita_right)))
        });
        group.bench_function("bn254_raw", |b| {
            b.iter(|| black_box(dot_product_raw(&bn_left, &bn_right)))
        });
        group.finish();
    }
}

fn bench_booleanity(c: &mut Criterion) {
    for (label, rows) in [("cache", 1 << 8), ("stream", 1 << 15)] {
        let akita = booleanity_input::<AkitaFp128>(rows, 10);
        let bn = booleanity_input::<Fr>(rows, 10);

        let expected = booleanity_eager(&akita);
        assert_eq!(booleanity_raw(&akita), expected);
        assert_eq!(booleanity_multi(&akita), expected);

        let mut group = c.benchmark_group(format!("booleanity/{label}"));
        group.throughput(Throughput::Elements((rows * BOOLEANITY_COLUMNS) as u64));
        group.bench_function(BenchmarkId::new("akita_raw", rows), |b| {
            b.iter(|| black_box(booleanity_raw(&akita)))
        });
        group.bench_function(BenchmarkId::new("akita_eager", rows), |b| {
            b.iter(|| black_box(booleanity_eager(&akita)))
        });
        group.bench_function(BenchmarkId::new("akita_solinas", rows), |b| {
            b.iter(|| black_box(booleanity_multi(&akita)))
        });
        group.bench_function(BenchmarkId::new("bn254_raw", rows), |b| {
            b.iter(|| black_box(booleanity_raw(&bn)))
        });
        group.finish();
    }
}

fn bench_d4(c: &mut Criterion) {
    for (label, rows) in [("cache", 1 << 8), ("stream", 1 << 15)] {
        let akita = d4_input::<AkitaFp128>(rows, 20);
        let bn = d4_input::<Fr>(rows, 20);

        let expected = d4_eager(&akita);
        assert_eq!(d4_raw(&akita), expected);
        assert_eq!(d4_multi(&akita), expected);
        assert_eq!(d4_multi(&bn), d4_raw(&bn));

        let mut group = c.benchmark_group(format!("d4/{label}"));
        group.throughput(Throughput::Elements((rows * D4_PRODUCTS) as u64));
        group.bench_function(BenchmarkId::new("akita_raw", rows), |b| {
            b.iter(|| black_box(d4_raw(&akita)))
        });
        group.bench_function(BenchmarkId::new("akita_eager", rows), |b| {
            b.iter(|| black_box(d4_eager(&akita)))
        });
        group.bench_function(BenchmarkId::new("akita_solinas", rows), |b| {
            b.iter(|| black_box(d4_multi(&akita)))
        });
        group.bench_function(BenchmarkId::new("bn254_raw", rows), |b| {
            b.iter(|| black_box(d4_raw(&bn)))
        });
        group.bench_function(BenchmarkId::new("bn254_multi", rows), |b| {
            b.iter(|| black_box(d4_multi(&bn)))
        });
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_dot_products, bench_booleanity, bench_d4
}
criterion_main!(benches);
