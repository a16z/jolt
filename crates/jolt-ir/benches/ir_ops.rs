#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, Fr};
use jolt_ir::{ExprBuilder, R1csVar};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

fn build_booleanity() -> (jolt_ir::Expr, Vec<Fr>, Vec<Fr>) {
    let b = ExprBuilder::new();
    let h = b.opening(0);
    let gamma = b.challenge(0);
    let expr = b.build(gamma * (h * h - h));
    let openings = vec![Fr::from_u64(42)];
    let challenges = vec![Fr::from_u64(7)];
    (expr, openings, challenges)
}

fn build_ram_style() -> (jolt_ir::Expr, Vec<Fr>, Vec<Fr>) {
    let b = ExprBuilder::new();
    let eq = b.opening(0);
    let ra = b.opening(1);
    let val = b.opening(2);
    let inc = b.opening(3);
    let gamma = b.challenge(0);
    let expr = b.build(eq * ra * val + gamma * eq * ra * inc);
    let mut rng = ChaCha8Rng::seed_from_u64(0xbeef);
    let openings: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let challenges = vec![Fr::random(&mut rng)];
    (expr, openings, challenges)
}

fn build_weighted_sum(n: usize) -> (jolt_ir::Expr, Vec<Fr>, Vec<Fr>) {
    let b = ExprBuilder::new();
    let mut acc = b.challenge(0) * b.opening(0);
    for i in 1..n {
        acc = acc + b.challenge(i as u32) * b.opening(i as u32);
    }
    let expr = b.build(acc);
    let mut rng = ChaCha8Rng::seed_from_u64(0xca5e);
    let openings: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let challenges: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    (expr, openings, challenges)
}

fn bench_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("Expr::evaluate");

    let (expr, o, ch) = build_booleanity();
    group.bench_function("booleanity", |bench| {
        bench.iter(|| black_box(&expr).evaluate::<Fr>(black_box(&o), black_box(&ch)));
    });

    let (expr, o, ch) = build_ram_style();
    group.bench_function("ram_style", |bench| {
        bench.iter(|| black_box(&expr).evaluate::<Fr>(black_box(&o), black_box(&ch)));
    });

    for n in [4, 16, 64] {
        let (expr, o, ch) = build_weighted_sum(n);
        group.bench_with_input(BenchmarkId::new("weighted_sum", n), &n, |bench, _| {
            bench.iter(|| black_box(&expr).evaluate::<Fr>(black_box(&o), black_box(&ch)));
        });
    }

    group.finish();
}

fn bench_to_sum_of_products(c: &mut Criterion) {
    let mut group = c.benchmark_group("Expr::to_sum_of_products");

    let (expr, _, _) = build_booleanity();
    group.bench_function("booleanity", |bench| {
        bench.iter(|| black_box(&expr).to_sum_of_products());
    });

    let (expr, _, _) = build_ram_style();
    group.bench_function("ram_style", |bench| {
        bench.iter(|| black_box(&expr).to_sum_of_products());
    });

    for n in [4, 16, 64] {
        let (expr, _, _) = build_weighted_sum(n);
        group.bench_with_input(BenchmarkId::new("weighted_sum", n), &n, |bench, _| {
            bench.iter(|| black_box(&expr).to_sum_of_products());
        });
    }

    group.finish();
}

fn bench_sop_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoP::evaluate");

    let (expr, o, ch) = build_booleanity();
    let sop = expr.to_sum_of_products();
    group.bench_function("booleanity", |bench| {
        bench.iter(|| black_box(&sop).evaluate::<Fr>(black_box(&o), black_box(&ch)));
    });

    let (expr, o, ch) = build_ram_style();
    let sop = expr.to_sum_of_products();
    group.bench_function("ram_style", |bench| {
        bench.iter(|| black_box(&sop).evaluate::<Fr>(black_box(&o), black_box(&ch)));
    });

    for n in [4, 16, 64] {
        let (expr, o, ch) = build_weighted_sum(n);
        let sop = expr.to_sum_of_products();
        group.bench_with_input(BenchmarkId::new("weighted_sum", n), &n, |bench, _| {
            bench.iter(|| black_box(&sop).evaluate::<Fr>(black_box(&o), black_box(&ch)));
        });
    }

    group.finish();
}

fn bench_emit_r1cs(c: &mut Criterion) {
    let mut group = c.benchmark_group("SoP::emit_r1cs");

    let (expr, _, _) = build_booleanity();
    let sop = expr.to_sum_of_products();
    let vars = [R1csVar(1)];
    let chal = [Fr::from_u64(7)];
    group.bench_function("booleanity", |bench| {
        bench.iter(|| {
            let mut next = 2u32;
            black_box(&sop).emit_r1cs::<Fr>(black_box(&vars), black_box(&chal), &mut next)
        });
    });

    let (expr, _, _) = build_ram_style();
    let sop = expr.to_sum_of_products();
    let vars: Vec<R1csVar> = (0..4).map(|i| R1csVar(i + 1)).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(0xbeef);
    let chal = [Fr::random(&mut rng)];
    group.bench_function("ram_style", |bench| {
        bench.iter(|| {
            let mut next = 5u32;
            black_box(&sop).emit_r1cs::<Fr>(black_box(&vars), black_box(&chal), &mut next)
        });
    });

    group.finish();
}

fn bench_fold_constants(c: &mut Criterion) {
    let b = ExprBuilder::new();
    let a = b.opening(0);
    let expr = b.build((b.constant(2) + b.constant(3)) * a * (b.constant(10) - b.constant(4)));

    c.bench_function("Expr::fold_constants", |bench| {
        bench.iter(|| black_box(&expr).fold_constants());
    });
}

fn bench_cse(c: &mut Criterion) {
    // (a+b)*(a+b) + (a+b)*(c+d) — shared (a+b) subtree
    let b = ExprBuilder::new();
    let a1 = b.opening(0);
    let b1 = b.opening(1);
    let a2 = b.opening(0);
    let b2 = b.opening(1);
    let a3 = b.opening(0);
    let b3 = b.opening(1);
    let c1 = b.opening(2);
    let d1 = b.opening(3);
    let expr = b.build((a1 + b1) * (a2 + b2) + (a3 + b3) * (c1 + d1));

    c.bench_function("Expr::eliminate_common_subexpressions", |bench| {
        bench.iter(|| black_box(&expr).eliminate_common_subexpressions());
    });
}

criterion_group!(
    benches,
    bench_evaluate,
    bench_to_sum_of_products,
    bench_sop_evaluate,
    bench_emit_r1cs,
    bench_fold_constants,
    bench_cse
);
criterion_main!(benches);
