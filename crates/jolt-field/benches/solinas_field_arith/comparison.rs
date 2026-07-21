use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup, Field, UniformRand};
use criterion::{black_box, Criterion};
use rand::{rngs::StdRng, SeedableRng};

pub(crate) fn bench_comparisons(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let inputs: Vec<BN254Fr> = (0..2048).map(|_| BN254Fr::rand(&mut rng)).collect();

    let mut group = c.benchmark_group("field_arith/comparison/bn254");

    group.bench_function("mul_add_chain_2048", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for x in inputs.iter() {
                acc = acc * x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_chain_2048", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for x in inputs.iter() {
                acc *= x;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_parallel_1024", |b| {
        b.iter(|| {
            let mut sum = BN254Fr::ZERO;
            for pair in inputs.chunks_exact(2) {
                sum += pair[0] * pair[1];
            }
            black_box(sum)
        })
    });

    group.bench_function("sqr_chain_2048", |b| {
        b.iter(|| {
            let mut acc = inputs[0];
            for _ in 0..2048 {
                acc.square_in_place();
            }
            black_box(acc)
        })
    });

    group.bench_function("inv_256", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for x in inputs[..256].iter() {
                acc *= x.inverse().unwrap_or(BN254Fr::ZERO);
            }
            black_box(acc)
        })
    });

    group.finish();
}
