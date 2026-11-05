//! The constant-time Gaussian sampler at Whiteout's production width.
//!
//! ```text
//! cargo bench -p jolt-crypto --features sample --bench gaussian
//! ```

#![expect(unused_results, clippy::unwrap_used)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use jolt_crypto::sample::{GaussianSampler, GaussianWidth};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

/// `setup.params().width()` of the production Whiteout parameters.
const PRODUCTION_K: u64 = 58_642_511_897;
/// One `fill_parallel` chunk.
const CHUNK: usize = 1 << 14;
/// fib 2^16's residual length.
const RESIDUAL: usize = 2_243_840;

fn bench_gaussian(c: &mut Criterion) {
    let sampler = GaussianSampler::new(GaussianWidth::from_multiplier(PRODUCTION_K).unwrap());
    let mut rng = ChaCha20Rng::seed_from_u64(0);

    let mut group = c.benchmark_group("gaussian");
    group.throughput(Throughput::Elements(CHUNK as u64));
    let mut chunk = vec![0i64; CHUNK];
    group.bench_function("fill_chunk", |b| {
        b.iter(|| sampler.fill(&mut rng, &mut chunk));
    });

    group.throughput(Throughput::Elements(RESIDUAL as u64));
    group.sample_size(20);
    let mut residual = vec![0i64; RESIDUAL];
    group.bench_function("fill_parallel_residual", |b| {
        b.iter(|| sampler.fill_parallel(&mut rng, &mut residual));
    });
    group.finish();
}

criterion_group!(benches, bench_gaussian);
criterion_main!(benches);
