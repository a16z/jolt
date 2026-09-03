#![expect(
    clippy::unwrap_used,
    reason = "benchmark setup and proof generation fail loudly"
)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_crypto::Bn254;
use jolt_field::{Field, Fr, Ring};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_zeromorph::ZeromorphScheme;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Scheme = ZeromorphScheme<Bn254>;

fn benchmark(c: &mut Criterion) {
    for num_vars in [20, 21] {
        let mut rng = ChaCha20Rng::seed_from_u64(num_vars as u64);
        let polynomial = Polynomial::random(num_vars, &mut rng);
        let points = (0..3)
            .map(|_| {
                (0..num_vars)
                    .map(|_| Fr::random(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let evaluations = points
            .iter()
            .map(|point| polynomial.evaluate(point))
            .collect::<Vec<_>>();
        let (pk, vk) = Scheme::setup_from_secret(
            Fr::from_u64(7),
            num_vars,
            Bn254::g1_generator(),
            Bn254::g2_generator(),
        )
        .unwrap();
        let commitment = Scheme::commit(&pk, polynomial.evaluations()).unwrap();
        let mut transcript = Blake2bTranscript::new(b"zeromorph-bench-proof");
        let proof = Scheme::open(
            &pk,
            polynomial.evaluations(),
            &points[0],
            evaluations[0],
            &mut transcript,
        )
        .unwrap();

        let mut group = c.benchmark_group(format!("zeromorph/2^{num_vars}"));
        let _ = group.sample_size(10);
        let _ = group.throughput(Throughput::Elements(1 << num_vars));
        let _ = group.bench_function(BenchmarkId::new("commit", num_vars), |b| {
            b.iter(|| Scheme::commit(&pk, polynomial.evaluations()).unwrap());
        });
        let _ = group.bench_function(BenchmarkId::new("open/1-point", num_vars), |b| {
            b.iter(|| {
                let mut transcript = Blake2bTranscript::new(b"zeromorph-bench-open-1");
                Scheme::open(
                    &pk,
                    polynomial.evaluations(),
                    &points[0],
                    evaluations[0],
                    &mut transcript,
                )
                .unwrap()
            });
        });
        let _ = group.bench_function(BenchmarkId::new("open/3-point", num_vars), |b| {
            b.iter(|| {
                let mut transcript = Blake2bTranscript::new(b"zeromorph-bench-open-3");
                Scheme::open_multi(
                    &pk,
                    polynomial.evaluations(),
                    &points,
                    &evaluations,
                    &mut transcript,
                )
                .unwrap()
            });
        });
        let _ = group.bench_function(BenchmarkId::new("verify", num_vars), |b| {
            b.iter(|| {
                let mut transcript = Blake2bTranscript::new(b"zeromorph-bench-proof");
                Scheme::verify(
                    &vk,
                    &commitment,
                    &points[0],
                    evaluations[0],
                    &proof,
                    &mut transcript,
                )
                .unwrap();
            });
        });
        group.finish();
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
