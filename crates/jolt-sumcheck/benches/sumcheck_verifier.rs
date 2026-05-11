#![expect(unused_results)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Fr, RandomSampling};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckProof, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use num_traits::One;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type F = Fr;

/// Build an honest sumcheck proof for a multilinear polynomial over `{0,1}^num_vars`
/// by running a minimal reference prover. The proof has degree-1 round polynomials.
///
/// Mirrors the `honest_prove` helper from the crate's unit tests so the verifier
/// is benchmarked against valid proofs that exercise every round, including the
/// final claim comparison and Fiat-Shamir absorption path.
fn honest_prove(
    evals: &[F],
    num_vars: usize,
    transcript: &mut Blake2bTranscript,
) -> SumcheckProof<F> {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = buf.len() / 2;
        let mut eval_0 = F::default();
        let mut eval_1 = F::default();
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }

        let c0 = eval_0;
        let c1 = eval_1 - eval_0;
        let round_poly = UnivariatePoly::new(vec![c0, c1]);

        for coeff in round_poly.coefficients() {
            coeff.append_to_transcript(transcript);
        }

        let r: F = transcript.challenge();
        round_polys.push(round_poly);

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    SumcheckProof {
        round_polynomials: round_polys,
    }
}

fn random_evals(num_vars: usize, seed: u64) -> Vec<F> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..(1usize << num_vars))
        .map(|_| F::random(&mut rng))
        .collect()
}

fn make_proof(num_vars: usize, seed: u64) -> (SumcheckClaim<F>, SumcheckProof<F>) {
    let evals = random_evals(num_vars, seed);
    let claimed_sum: F = evals.iter().copied().sum();
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-bench");
    let proof = honest_prove(&evals, num_vars, &mut transcript);
    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };
    (claim, proof)
}

fn bench_single_verifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckVerifier::verify");
    for num_vars in [8, 14, 18, 22] {
        let (claim, proof) = make_proof(num_vars, 1000 + num_vars as u64);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter(|| {
                    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-bench");
                    let result = SumcheckVerifier::verify(
                        black_box(&claim),
                        black_box(&proof.round_polynomials),
                        &mut transcript,
                    );
                    assert!(result.is_ok());
                });
            },
        );
    }
    group.finish();
}

fn bench_batched_verifier_same_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("BatchedSumcheckVerifier::verify/same_size");

    for &(num_vars, n_claims) in &[(14usize, 2usize), (14, 4), (18, 2), (18, 4)] {
        let evals_per_claim: Vec<Vec<F>> = (0..n_claims)
            .map(|i| random_evals(num_vars, 2000 + i as u64 + num_vars as u64))
            .collect();
        let sums: Vec<F> = evals_per_claim
            .iter()
            .map(|evals| evals.iter().copied().sum())
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"jolt-sumcheck-batched-bench");
        for s in &sums {
            s.append_to_transcript(&mut prover_transcript);
        }
        let alpha: F = prover_transcript.challenge();
        let mut alpha_pow = F::one();
        let mut combined = vec![F::default(); 1usize << num_vars];
        for evals in &evals_per_claim {
            for (slot, e) in combined.iter_mut().zip(evals) {
                *slot += alpha_pow * *e;
            }
            alpha_pow *= alpha;
        }
        let proof = honest_prove(&combined, num_vars, &mut prover_transcript);

        let claims: Vec<SumcheckClaim<F>> = sums
            .iter()
            .map(|&s| SumcheckClaim {
                num_vars,
                degree: 1,
                claimed_sum: s,
            })
            .collect();

        let label = format!("n_vars={num_vars}/n_claims={n_claims}");
        group.bench_with_input(BenchmarkId::from_parameter(label), &num_vars, |bench, _| {
            bench.iter(|| {
                let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-batched-bench");
                let result = BatchedSumcheckVerifier::verify(
                    black_box(&claims),
                    black_box(&proof.round_polynomials),
                    &mut transcript,
                );
                assert!(result.is_ok());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_verifier,
    bench_batched_verifier_same_size
);
criterion_main!(benches);
