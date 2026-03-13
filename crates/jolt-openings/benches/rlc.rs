#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use jolt_openings::{rlc_combine, rlc_combine_scalars};

fn bench_rlc_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("rlc_combine");
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let rho: Fr = Field::random(&mut rng);

    for (n_polys, num_vars) in [(8, 18), (32, 14)] {
        let len = 1 << num_vars;
        let tables: Vec<Vec<Fr>> = (0..n_polys)
            .map(|_| (0..len).map(|_| Fr::random(&mut rng)).collect())
            .collect();
        let refs: Vec<&[Fr]> = tables.iter().map(|t| t.as_slice()).collect();

        group.bench_with_input(
            BenchmarkId::new(format!("{n_polys}×2^{num_vars}"), ""),
            &(),
            |bench, ()| {
                bench.iter(|| rlc_combine(black_box(&refs), black_box(rho)));
            },
        );
    }
    group.finish();
}

fn bench_rlc_combine_scalars(c: &mut Criterion) {
    let mut group = c.benchmark_group("rlc_combine_scalars");
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let rho: Fr = Field::random(&mut rng);

    for n in [8, 32] {
        let evals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| rlc_combine_scalars(black_box(&evals), black_box(rho)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rlc_combine, bench_rlc_combine_scalars);
criterion_main!(benches);
