use criterion::{black_box, criterion_group, criterion_main, Criterion};
use liblasso::dense_ml_poly_bench;

fn bench_dense_mlpoly_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMLPoly Evaluation");
    dense_ml_poly_bench(&mut group);
    group.finish();
}

criterion_group!(benches, bench_dense_mlpoly_evaluate);
criterion_main!(benches);
