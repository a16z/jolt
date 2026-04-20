#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type TestScheme = HyperKZGScheme<Bn254>;

fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xbe0c);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = TestScheme::setup(&mut rng, max_degree, g1, g2);
    let vk = TestScheme::verifier_setup(&pk);
    (pk, vk)
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperkzg_commit");
    for num_vars in [8, 10, 12, 14] {
        let n = 1 << num_vars;
        let (pk, _) = make_setup(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &nv| {
                b.iter_batched(
                    || {
                        let mut rng = ChaCha20Rng::seed_from_u64(0);
                        Polynomial::<Fr>::random(nv, &mut rng)
                    },
                    |poly| TestScheme::commit(poly.evaluations(), &pk),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperkzg_open");
    for num_vars in [8, 10, 12, 14] {
        let n = 1 << num_vars;
        let (pk, _) = make_setup(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &nv| {
                b.iter_batched(
                    || {
                        let mut rng = ChaCha20Rng::seed_from_u64(0);
                        let poly = Polynomial::<Fr>::random(nv, &mut rng);
                        let point: Vec<Fr> =
                            (0..nv).map(|_| <Fr as Field>::random(&mut rng)).collect();
                        let eval = poly.evaluate(&point);
                        (poly, point, eval)
                    },
                    |(poly, point, eval)| {
                        let mut transcript = jolt_transcript::Blake2bTranscript::new(b"bench-open");
                        <TestScheme as CommitmentScheme>::open(
                            &poly,
                            &point,
                            eval,
                            &pk,
                            None,
                            &mut transcript,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperkzg_verify");
    for num_vars in [8, 10, 12, 14] {
        let n = 1 << num_vars;
        let (pk, vk) = make_setup(n);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &nv| {
                b.iter_batched(
                    || {
                        let mut rng = ChaCha20Rng::seed_from_u64(0);
                        let poly = Polynomial::<Fr>::random(nv, &mut rng);
                        let point: Vec<Fr> =
                            (0..nv).map(|_| <Fr as Field>::random(&mut rng)).collect();
                        let eval = poly.evaluate(&point);
                        let (commitment, ()) = TestScheme::commit(poly.evaluations(), &pk);
                        let mut transcript =
                            jolt_transcript::Blake2bTranscript::new(b"bench-verify");
                        let proof = <TestScheme as CommitmentScheme>::open(
                            &poly,
                            &point,
                            eval,
                            &pk,
                            None,
                            &mut transcript,
                        );
                        (commitment, point, eval, proof)
                    },
                    |(commitment, point, eval, proof)| {
                        let mut transcript =
                            jolt_transcript::Blake2bTranscript::new(b"bench-verify");
                        <TestScheme as CommitmentScheme>::verify(
                            &commitment,
                            &point,
                            eval,
                            &proof,
                            &vk,
                            &mut transcript,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperkzg_combine");
    for count in [2, 4, 8, 16] {
        let num_vars = 10;
        let n = 1 << num_vars;
        let (pk, _) = make_setup(n);
        let mut rng = ChaCha20Rng::seed_from_u64(0);

        let commitments: Vec<_> = (0..count)
            .map(|_| {
                let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
                let (c, ()) = TestScheme::commit(poly.evaluations(), &pk);
                c
            })
            .collect();
        let scalars: Vec<Fr> = (0..count).map(|_| Fr::random(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| TestScheme::combine(&commitments, &scalars));
        });
    }
    group.finish();
}

fn bench_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperkzg_setup");
    for num_vars in [8, 10, 12] {
        let n = 1 << num_vars;
        group.bench_with_input(BenchmarkId::from_parameter(num_vars), &num_vars, |b, _| {
            b.iter(|| {
                let mut rng = ChaCha20Rng::seed_from_u64(0xbe0c);
                let g1 = Bn254::g1_generator();
                let g2 = Bn254::g2_generator();
                TestScheme::setup(&mut rng, n, g1, g2)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_setup,
    bench_commit,
    bench_open,
    bench_verify,
    bench_combine,
);
criterion_main!(benches);
