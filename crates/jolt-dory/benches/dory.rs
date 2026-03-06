#![allow(unused_results)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use jolt_dory::{DoryScheme, DoryVerifierSetup};
use jolt_field::{Field, Fr};
use jolt_openings::{CommitmentScheme, StreamingCommitment};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn bench_setup_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_prover");
    for num_vars in [4, 8, 12] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &nv| {
                b.iter(|| DoryScheme::setup_prover(nv));
            },
        );
    }
    group.finish();
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit");
    for num_vars in [4, 8, 12] {
        let setup = DoryScheme::setup_prover(num_vars);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &nv| {
                b.iter_batched(
                    || {
                        let mut rng = ChaCha20Rng::seed_from_u64(0);
                        Polynomial::<Fr>::random(nv, &mut rng)
                    },
                    |poly| DoryScheme::commit(poly.evaluations(), &setup),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("open");
    for num_vars in [4, 8, 12] {
        let setup = DoryScheme::setup_prover(num_vars);
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
                        let mut transcript =
                            jolt_transcript::Blake2bTranscript::new(b"bench-open");
                        DoryScheme::open(
                            &poly,
                            &point,
                            eval,
                            &setup,
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
    let mut group = c.benchmark_group("verify");
    for num_vars in [4, 8, 12] {
        let setup = DoryScheme::setup_prover(num_vars);
        let verifier_setup = DoryVerifierSetup(setup.0.to_verifier_setup());
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
                        let (commitment, _) = DoryScheme::commit(poly.evaluations(), &setup);
                        let mut transcript =
                            jolt_transcript::Blake2bTranscript::new(b"bench-verify");
                        let proof = DoryScheme::open(
                            &poly,
                            &point,
                            eval,
                            &setup,
                            None,
                            &mut transcript,
                        );
                        (commitment, point, eval, proof)
                    },
                    |(commitment, point, eval, proof)| {
                        let mut transcript =
                            jolt_transcript::Blake2bTranscript::new(b"bench-verify");
                        DoryScheme::verify(
                            &commitment,
                            &point,
                            eval,
                            &proof,
                            &verifier_setup,
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

fn bench_streaming_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_commit");
    for num_vars in [4, 8, 12] {
        let setup = DoryScheme::setup_prover(num_vars);
        let sigma = num_vars.div_ceil(2);
        let num_cols = 1usize << sigma;

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, &nv| {
                b.iter_batched(
                    || {
                        let mut rng = ChaCha20Rng::seed_from_u64(0);
                        Polynomial::<Fr>::random(nv, &mut rng)
                    },
                    |poly| {
                        let mut partial = DoryScheme::begin(&setup);
                        for row in poly.evaluations().chunks(num_cols) {
                            DoryScheme::feed(&mut partial, row, &setup);
                        }
                        DoryScheme::finish(partial, &setup)
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_combine(c: &mut Criterion) {
    let mut group = c.benchmark_group("combine");
    for num_vars in [4, 8] {
        let setup = DoryScheme::setup_prover(num_vars);
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
        let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);
        let (commit_a, _) = DoryScheme::commit(poly_a.evaluations(), &setup);
        let (commit_b, _) = DoryScheme::commit(poly_b.evaluations(), &setup);
        let s_a = Fr::random(&mut rng);
        let s_b = Fr::random(&mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |b, _| {
                b.iter(|| {
                    <DoryScheme as jolt_openings::AdditivelyHomomorphic>::combine(
                        &[commit_a.clone(), commit_b.clone()],
                        &[s_a, s_b],
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_setup_prover,
    bench_commit,
    bench_open,
    bench_verify,
    bench_streaming_commit,
    bench_combine,
);
criterion_main!(benches);
