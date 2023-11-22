use ark_curve25519::Fr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use liblasso::{poly::dense_mlpoly::DensePolynomial, utils::gen_random_point};

fn bench_dense_mlpoly_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMLPoly Evaluation");
    let evals: Vec<Fr> = gen_random_point::<Fr>(1 << 10);
    let poly = DensePolynomial::new(evals.clone()); 

    let r: Vec<Fr> = gen_random_point::<Fr>(10);

    group.bench_function("evaluate", |b| {
        b.iter(|| {
            let result = black_box(poly.evaluate(&r));
            criterion::black_box(result);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_dense_mlpoly_evaluate);
criterion_main!(benches);
