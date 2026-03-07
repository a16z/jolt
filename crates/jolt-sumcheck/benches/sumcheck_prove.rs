#![allow(unused_results)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::{Field, Fr};
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::batched::BatchedSumcheckProver;
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prover::{SumcheckCompute, SumcheckProver};
use jolt_sumcheck::verifier::SumcheckVerifier;
use jolt_transcript::{Blake2bTranscript, Transcript};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn challenge_to_field(c: u128) -> Fr {
    Fr::from_u128(c)
}

struct EqProductWitness {
    poly: Polynomial<Fr>,
    eq_evals: Vec<Fr>,
}

impl EqProductWitness {
    fn new(poly: Polynomial<Fr>, tau: &[Fr]) -> Self {
        let eq_evals = EqPolynomial::new(tau.to_vec()).evaluations();
        Self { poly, eq_evals }
    }
}

impl SumcheckCompute<Fr> for EqProductWitness {
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.poly.len() / 2;
        let mut evals = [Fr::zero(); 3];
        for i in 0..half {
            let f_lo = self.poly.evaluations()[i];
            let f_hi = self.poly.evaluations()[i + half];
            let eq_lo = self.eq_evals[i];
            let eq_hi = self.eq_evals[i + half];
            evals[0] += f_lo * eq_lo;
            evals[1] += f_hi * eq_hi;
            let f_at_2 = f_lo + (f_hi - f_lo) + (f_hi - f_lo);
            let eq_at_2 = eq_lo + (eq_hi - eq_lo) + (eq_hi - eq_lo);
            evals[2] += f_at_2 * eq_at_2;
        }
        let points: Vec<(Fr, Fr)> = evals
            .iter()
            .enumerate()
            .map(|(t, &y)| (Fr::from_u64(t as u64), y))
            .collect();
        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: Fr) {
        self.poly.bind(challenge);
        let half = self.eq_evals.len() / 2;
        for i in 0..half {
            let lo = self.eq_evals[i];
            let hi = self.eq_evals[i + half];
            self.eq_evals[i] = lo + challenge * (hi - lo);
        }
        self.eq_evals.truncate(half);
    }
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckProver::prove");
    for num_vars in [14, 18] {
        let mut rng = ChaCha20Rng::seed_from_u64(num_vars as u64);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let claimed_sum = poly.evaluate(&tau);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let _ = group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter_batched(
                    || {
                        (
                            EqProductWitness::new(poly.clone(), &tau),
                            Blake2bTranscript::new(b"bench"),
                        )
                    },
                    |(mut witness, mut transcript)| {
                        SumcheckProver::prove(
                            black_box(&claim),
                            &mut witness,
                            &mut transcript,
                            challenge_to_field,
                        )
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("SumcheckVerifier::verify");
    for num_vars in [14, 18] {
        let mut rng = ChaCha20Rng::seed_from_u64(100 + num_vars as u64);
        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let claimed_sum = poly.evaluate(&tau);

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let mut witness = EqProductWitness::new(poly, &tau);
        let mut pt = Blake2bTranscript::new(b"bench");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt, challenge_to_field);

        let _ = group.bench_with_input(
            BenchmarkId::from_parameter(num_vars),
            &num_vars,
            |bench, _| {
                bench.iter_batched(
                    || Blake2bTranscript::new(b"bench"),
                    |mut transcript| {
                        SumcheckVerifier::verify(
                            black_box(&claim),
                            black_box(&proof),
                            &mut transcript,
                            challenge_to_field,
                        )
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

struct PlainSumWitness {
    poly: Polynomial<Fr>,
}

impl SumcheckCompute<Fr> for PlainSumWitness {
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.poly.len() / 2;
        let mut sum_lo = Fr::zero();
        let mut sum_hi = Fr::zero();
        for i in 0..half {
            sum_lo += self.poly.evaluations()[i];
            sum_hi += self.poly.evaluations()[i + half];
        }
        UnivariatePoly::new(vec![sum_lo, sum_hi - sum_lo])
    }

    fn bind(&mut self, challenge: Fr) {
        self.poly.bind(challenge);
    }
}

fn bench_batched_prove(c: &mut Criterion) {
    let num_vars = 14;
    let num_claims = 8;
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    let polys: Vec<Polynomial<Fr>> = (0..num_claims)
        .map(|_| Polynomial::random(num_vars, &mut rng))
        .collect();
    let sums: Vec<Fr> = polys
        .iter()
        .map(|p| p.evaluations().iter().copied().sum())
        .collect();

    let claims: Vec<SumcheckClaim<Fr>> = sums
        .iter()
        .map(|&s| SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: s,
        })
        .collect();

    let _ = c.bench_function("BatchedSumcheck::prove/8×14vars", |bench| {
        bench.iter_batched(
            || {
                let witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = polys
                    .iter()
                    .map(|p| -> Box<dyn SumcheckCompute<Fr>> {
                        Box::new(PlainSumWitness { poly: p.clone() })
                    })
                    .collect();
                (witnesses, Blake2bTranscript::new(b"bench"))
            },
            |(mut witnesses, mut transcript)| {
                BatchedSumcheckProver::prove(
                    black_box(&claims),
                    &mut witnesses,
                    &mut transcript,
                    challenge_to_field,
                )
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, bench_prove, bench_verify, bench_batched_prove);
criterion_main!(benches);
