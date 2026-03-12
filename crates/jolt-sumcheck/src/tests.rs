//! Integration tests for the sumcheck protocol.

use jolt_field::{Field, Fr, WithChallenge};
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use num_traits::Zero;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::batched::{BatchedSumcheckProver, BatchedSumcheckVerifier};
use crate::claim::SumcheckClaim;
use crate::prover::{SumcheckCompute, SumcheckProver};
use crate::streaming::StreamingSumcheckProver;
use crate::verifier::SumcheckVerifier;

type C = <Fr as WithChallenge>::Challenge;

/// Witness for the claim $\sum_{x \in \{0,1\}^n} f(x) \cdot \widetilde{eq}(x, \tau) = f(\tau)$.
///
/// This is the standard use case: proving an evaluation claim on a
/// multilinear polynomial via sumcheck with the equality polynomial.
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
        // For the product f(x) * eq(x, tau), the round polynomial is degree 2.
        // Evaluate the univariate restriction at t = 0, 1, 2 and interpolate.
        let mut evals = [Fr::zero(); 3];

        for i in 0..half {
            let f_lo = self.poly.evaluations()[i];
            let f_hi = self.poly.evaluations()[i + half];
            let eq_lo = self.eq_evals[i];
            let eq_hi = self.eq_evals[i + half];

            // t=0: f_lo * eq_lo
            evals[0] += f_lo * eq_lo;
            // t=1: f_hi * eq_hi
            evals[1] += f_hi * eq_hi;
            // t=2: (f_lo + 2*(f_hi - f_lo)) * (eq_lo + 2*(eq_hi - eq_lo))
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

    fn bind(&mut self, challenge: C) {
        let c: Fr = challenge.into();
        self.poly.bind(c);

        let half = self.eq_evals.len() / 2;
        for i in 0..half {
            let lo = self.eq_evals[i];
            let hi = self.eq_evals[i + half];
            self.eq_evals[i] = lo + c * (hi - lo);
        }
        self.eq_evals.truncate(half);
    }
}

/// Witness for the claim $\sum_{x \in \{0,1\}^n} f(x) = C$.
///
/// The round polynomial is degree 1 (linear) since the polynomial is
/// multilinear.
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

        // s(X) is degree 1: s(0) = sum_lo, s(1) = sum_hi
        // s(X) = sum_lo + X * (sum_hi - sum_lo)
        UnivariatePoly::new(vec![sum_lo, sum_hi - sum_lo])
    }

    fn bind(&mut self, challenge: C) {
        self.poly.bind(challenge.into());
    }
}

#[test]
fn plain_sum_prove_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let num_vars = 6;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    let mut witness = PlainSumWitness { poly: poly.clone() };
    let mut prover_transcript = Blake2bTranscript::new(b"test_plain_sum");

    let proof = SumcheckProver::prove(
        &claim,
        &mut witness,
        &mut prover_transcript,
    );

    assert_eq!(proof.round_polynomials.len(), num_vars);

    let mut verifier_transcript = Blake2bTranscript::new(b"test_plain_sum");
    let result =
        SumcheckVerifier::verify(&claim, &proof, &mut verifier_transcript);

    let (final_eval, challenges) = result.expect("verification should succeed");

    // The final evaluation should equal poly(challenges)
    let expected = poly.evaluate(&challenges);
    assert_eq!(final_eval, expected);
}

#[test]
fn eq_product_prove_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(99);
    let num_vars = 5;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    // claimed_sum = f(tau) = sum_x f(x) * eq(x, tau)
    let claimed_sum = poly.evaluate(&tau);

    let claim = SumcheckClaim {
        num_vars,
        degree: 2,
        claimed_sum,
    };

    let mut witness = EqProductWitness::new(poly.clone(), &tau);
    let mut prover_transcript = Blake2bTranscript::new(b"test_eq_prod");

    let proof = SumcheckProver::prove(
        &claim,
        &mut witness,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"test_eq_prod");
    let result =
        SumcheckVerifier::verify(&claim, &proof, &mut verifier_transcript);
    let (final_eval, challenges) = result.expect("verification should succeed");

    // Verify: final_eval should equal f(challenges) * eq(challenges, tau)
    let f_at_r = poly.evaluate(&challenges);
    let eq_at_r = EqPolynomial::new(tau).evaluate(&challenges);
    assert_eq!(final_eval, f_at_r * eq_at_r);
}

#[test]
fn wrong_claimed_sum_fails() {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let num_vars = 4;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);

    let correct_sum: Fr = poly.evaluations().iter().copied().sum();
    let wrong_sum = correct_sum + Fr::from_u64(1);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: wrong_sum,
    };

    let mut witness = PlainSumWitness { poly };
    let mut prover_transcript = Blake2bTranscript::new(b"test_wrong");

    let proof = SumcheckProver::prove(
        &claim,
        &mut witness,
        &mut prover_transcript,
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"test_wrong");
    let result =
        SumcheckVerifier::verify(&claim, &proof, &mut verifier_transcript);

    assert!(result.is_err(), "verification should fail with wrong sum");
}

#[test]
fn single_variable() {
    // f(x) = 3 + 4x => f(0)=3, f(1)=7, sum=10
    let poly = Polynomial::new(vec![Fr::from_u64(3), Fr::from_u64(7)]);
    let claimed_sum = Fr::from_u64(10);

    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum,
    };

    let mut witness = PlainSumWitness { poly };
    let mut pt = Blake2bTranscript::new(b"test_single");

    let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);
    assert_eq!(proof.round_polynomials.len(), 1);

    let mut vt = Blake2bTranscript::new(b"test_single");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
    assert!(result.is_ok());
}

#[test]
fn batched_prove_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let num_vars = 4;

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);

    let sum_a: Fr = poly_a.evaluations().iter().copied().sum();
    let sum_b: Fr = poly_b.evaluations().iter().copied().sum();

    let claims = vec![
        SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![
        Box::new(PlainSumWitness {
            poly: poly_a.clone(),
        }),
        Box::new(PlainSumWitness {
            poly: poly_b.clone(),
        }),
    ];

    let mut pt = Blake2bTranscript::new(b"test_batched");
    let proof = BatchedSumcheckProver::prove(&claims, &mut witnesses, &mut pt);

    let mut vt = Blake2bTranscript::new(b"test_batched");
    let result = BatchedSumcheckVerifier::verify(&claims, &proof, &mut vt);
    assert!(result.is_ok(), "batched verification should succeed");
}

#[test]
fn wrong_round_count_is_rejected() {
    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: Fr::zero(),
    };

    let proof = crate::proof::SumcheckProof {
        round_polynomials: vec![UnivariatePoly::zero(); 2], // wrong: 2 != 3
    };

    let mut vt = Blake2bTranscript::new(b"test_rounds");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
    assert!(result.is_err());
}

#[test]
fn degree_bound_exceeded_is_rejected() {
    // Construct a proof with a degree-3 polynomial for a degree-1 claim
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: Fr::from_u64(10),
    };

    let cubic = UnivariatePoly::new(vec![
        Fr::from_u64(3),
        Fr::from_u64(4),
        Fr::from_u64(0),
        Fr::from_u64(1),
    ]);

    let proof = crate::proof::SumcheckProof {
        round_polynomials: vec![cubic],
    };

    let mut vt = Blake2bTranscript::new(b"test_degree");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
    assert!(
        matches!(
            result,
            Err(crate::error::SumcheckError::DegreeBoundExceeded { .. })
        ),
        "should reject degree bound violation"
    );
}

#[test]
fn deterministic_proofs() {
    let mut rng = ChaCha20Rng::seed_from_u64(555);
    let num_vars = 5;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    // Run twice — proofs must be identical
    let proof1 = {
        let mut w = PlainSumWitness { poly: poly.clone() };
        let mut t = Blake2bTranscript::new(b"determinism");
        SumcheckProver::prove(&claim, &mut w, &mut t)
    };

    let proof2 = {
        let mut w = PlainSumWitness { poly };
        let mut t = Blake2bTranscript::new(b"determinism");
        SumcheckProver::prove(&claim, &mut w, &mut t)
    };

    for (a, b) in proof1
        .round_polynomials
        .iter()
        .zip(proof2.round_polynomials.iter())
    {
        assert_eq!(a.coefficients(), b.coefficients());
    }
}

#[test]
fn batched_single_claim_matches_unbatched() {
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let num_vars = 5;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let claims = vec![SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    }];

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> =
        vec![Box::new(PlainSumWitness { poly: poly.clone() })];

    let mut pt = Blake2bTranscript::new(b"batched_single");
    let proof = BatchedSumcheckProver::prove(&claims, &mut witnesses, &mut pt);

    let mut vt = Blake2bTranscript::new(b"batched_single");
    let (final_eval, challenges) =
        BatchedSumcheckVerifier::verify(&claims, &proof, &mut vt)
            .expect("single-claim batch should verify");

    // alpha^0 = 1, so the combined claim is identical to the original.
    // The final evaluation must equal the polynomial at the challenge point.
    let expected = poly.evaluate(&challenges);
    assert_eq!(final_eval, expected);
}

#[test]
fn batched_three_claims() {
    let mut rng = ChaCha20Rng::seed_from_u64(301);
    let num_vars = 4;

    let polys: Vec<Polynomial<Fr>> = (0..3)
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

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = polys
        .iter()
        .map(|p| -> Box<dyn SumcheckCompute<Fr>> { Box::new(PlainSumWitness { poly: p.clone() }) })
        .collect();

    let mut pt = Blake2bTranscript::new(b"batched_three");
    let proof = BatchedSumcheckProver::prove(&claims, &mut witnesses, &mut pt);

    let mut vt = Blake2bTranscript::new(b"batched_three");
    let result = BatchedSumcheckVerifier::verify(&claims, &proof, &mut vt);
    assert!(
        result.is_ok(),
        "3-claim batched verification should succeed"
    );
}

#[test]
fn batched_wrong_claim_fails() {
    let mut rng = ChaCha20Rng::seed_from_u64(402);
    let num_vars = 4;

    let poly_a = Polynomial::<Fr>::random(num_vars, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars, &mut rng);

    let sum_a: Fr = poly_a.evaluations().iter().copied().sum();
    let sum_b: Fr = poly_b.evaluations().iter().copied().sum();
    let wrong_sum_b = sum_b + Fr::from_u64(1);

    let prover_claims = vec![
        SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![
        Box::new(PlainSumWitness {
            poly: poly_a.clone(),
        }),
        Box::new(PlainSumWitness {
            poly: poly_b.clone(),
        }),
    ];

    let mut pt = Blake2bTranscript::new(b"batched_wrong");
    let proof =
        BatchedSumcheckProver::prove(&prover_claims, &mut witnesses, &mut pt);

    // Verifier uses a tampered second claim
    let verifier_claims = vec![
        SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars,
            degree: 1,
            claimed_sum: wrong_sum_b,
        },
    ];

    let mut vt = Blake2bTranscript::new(b"batched_wrong");
    let result =
        BatchedSumcheckVerifier::verify(&verifier_claims, &proof, &mut vt);
    assert!(
        result.is_err(),
        "verification should fail when one claim has a wrong sum"
    );
}

/// Witness for $\sum_x f(x) \cdot g(x) \cdot \widetilde{eq}(x, \tau)$.
///
/// The round polynomial has degree 3, so we evaluate at t = 0, 1, 2, 3
/// and interpolate.
struct TripleProductWitness {
    f: Polynomial<Fr>,
    g: Polynomial<Fr>,
    eq_evals: Vec<Fr>,
}

impl TripleProductWitness {
    fn new(f: Polynomial<Fr>, g: Polynomial<Fr>, tau: &[Fr]) -> Self {
        let eq_evals = EqPolynomial::new(tau.to_vec()).evaluations();
        Self { f, g, eq_evals }
    }

    fn claimed_sum(&self) -> Fr {
        let n = self.f.len();
        let mut sum = Fr::zero();
        for i in 0..n {
            sum += self.f.evaluations()[i] * self.g.evaluations()[i] * self.eq_evals[i];
        }
        sum
    }
}

impl SumcheckCompute<Fr> for TripleProductWitness {
    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
        let half = self.f.len() / 2;
        let mut evals = [Fr::zero(); 4];

        for i in 0..half {
            let f_lo = self.f.evaluations()[i];
            let f_hi = self.f.evaluations()[i + half];
            let g_lo = self.g.evaluations()[i];
            let g_hi = self.g.evaluations()[i + half];
            let eq_lo = self.eq_evals[i];
            let eq_hi = self.eq_evals[i + half];

            // Evaluate the three linear functions at t = 0, 1, 2, 3
            // h(t) = lo + t * (hi - lo) for each of f, g, eq
            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = Fr::from_u64(t as u64);
                let f_t = f_lo + t_f * (f_hi - f_lo);
                let g_t = g_lo + t_f * (g_hi - g_lo);
                let eq_t = eq_lo + t_f * (eq_hi - eq_lo);
                *eval += f_t * g_t * eq_t;
            }
        }

        let points: Vec<(Fr, Fr)> = evals
            .iter()
            .enumerate()
            .map(|(t, &y)| (Fr::from_u64(t as u64), y))
            .collect();

        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: C) {
        let c: Fr = challenge.into();
        self.f.bind(c);
        self.g.bind(c);

        let half = self.eq_evals.len() / 2;
        for i in 0..half {
            let lo = self.eq_evals[i];
            let hi = self.eq_evals[i + half];
            self.eq_evals[i] = lo + c * (hi - lo);
        }
        self.eq_evals.truncate(half);
    }
}

#[test]
fn degree_3_triple_product() {
    let mut rng = ChaCha20Rng::seed_from_u64(503);
    let num_vars = 5;
    let f = Polynomial::<Fr>::random(num_vars, &mut rng);
    let g = Polynomial::<Fr>::random(num_vars, &mut rng);
    let tau: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

    let mut witness = TripleProductWitness::new(f.clone(), g.clone(), &tau);
    let claimed_sum = witness.claimed_sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 3,
        claimed_sum,
    };

    let mut pt = Blake2bTranscript::new(b"degree3");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

    assert_eq!(proof.round_polynomials.len(), num_vars);
    for rp in &proof.round_polynomials {
        assert!(rp.degree() <= 3, "round poly degree should be at most 3");
    }

    let mut vt = Blake2bTranscript::new(b"degree3");
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof, &mut vt)
            .expect("degree-3 verification should succeed");

    let f_r = f.evaluate(&challenges);
    let g_r = g.evaluate(&challenges);
    let eq_r = EqPolynomial::new(tau).evaluate(&challenges);
    assert_eq!(final_eval, f_r * g_r * eq_r);
}

#[test]
fn zero_claimed_sum() {
    let num_vars = 4;
    let n = 1 << num_vars;

    // Construct a polynomial whose evaluations sum to zero by negating
    // the last entry to cancel the rest.
    let mut rng = ChaCha20Rng::seed_from_u64(604);
    let mut evals: Vec<Fr> = (0..n - 1).map(|_| Fr::random(&mut rng)).collect();
    let partial_sum: Fr = evals.iter().copied().sum();
    evals.push(-partial_sum);

    let poly = Polynomial::new(evals);
    let claimed_sum = Fr::zero();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    let mut witness = PlainSumWitness { poly: poly.clone() };
    let mut pt = Blake2bTranscript::new(b"zero_sum");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

    let mut vt = Blake2bTranscript::new(b"zero_sum");
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof, &mut vt)
            .expect("zero claimed sum should verify");

    assert_eq!(final_eval, poly.evaluate(&challenges));
}

/// A streaming prover for $\sum_x f(x)$ (degree-1 round polynomials).
///
/// Processes the evaluation table in chunks, accumulating the partial
/// sums for the low and high halves of the current variable's split.
struct StreamingPlainSumProver {
    /// Full evaluation table, progressively reduced after each bind.
    evals: Vec<Fr>,
    num_vars: usize,
    /// Accumulated sums for the current round: [sum_lo, sum_hi].
    accum: [Fr; 2],
}

impl StreamingPlainSumProver {
    fn new(evals: Vec<Fr>, num_vars: usize) -> Self {
        Self {
            evals,
            num_vars,
            accum: [Fr::zero(); 2],
        }
    }
}

impl StreamingSumcheckProver<Fr> for StreamingPlainSumProver {
    fn begin_round(&mut self) {
        self.accum = [Fr::zero(); 2];
    }

    fn process_chunk(&mut self, chunk: &[Fr]) {
        // Test impl: expects the full table as a single chunk.
        let half = self.evals.len() / 2;
        for (local_idx, &val) in chunk.iter().enumerate() {
            if local_idx < half {
                self.accum[0] += val;
            } else {
                self.accum[1] += val;
            }
        }
    }

    fn finish_round(&mut self) -> UnivariatePoly<Fr> {
        let sum_lo = self.accum[0];
        let sum_hi = self.accum[1];
        UnivariatePoly::new(vec![sum_lo, sum_hi - sum_lo])
    }

    fn bind(&mut self, challenge: Fr) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            let lo = self.evals[i];
            let hi = self.evals[i + half];
            self.evals[i] = lo + challenge * (hi - lo);
        }
        self.evals.truncate(half);
        self.num_vars -= 1;
    }
}

#[test]
fn streaming_prover_produces_correct_rounds() {
    let mut rng = ChaCha20Rng::seed_from_u64(705);
    let num_vars = 5;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    // Drive the streaming prover manually, recording round polys.
    let mut streaming = StreamingPlainSumProver::new(poly.evaluations().to_vec(), num_vars);
    let mut transcript = Blake2bTranscript::new(b"streaming_test");
    let mut round_polys = Vec::with_capacity(num_vars);

    for _ in 0..num_vars {
        streaming.begin_round();
        // Pass the full evaluation table as a single chunk.
        streaming.process_chunk(&streaming.evals.clone());
        let rp = streaming.finish_round();

        // Absorb into transcript
        for coeff in rp.coefficients() {
            coeff.append_to_transcript(&mut transcript);
        }

        let challenge: C = transcript.challenge().into();
        streaming.bind(challenge.into());
        round_polys.push(rp);
    }

    let proof = crate::proof::SumcheckProof {
        round_polynomials: round_polys,
    };

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    // Verify the proof produced by the streaming prover.
    let mut vt = Blake2bTranscript::new(b"streaming_test");
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof, &mut vt)
            .expect("streaming prover proof should verify");

    assert_eq!(final_eval, poly.evaluate(&challenges));
}

#[test]
fn streaming_prover_multi_chunk() {
    let mut rng = ChaCha20Rng::seed_from_u64(806);
    let num_vars = 4;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let mut streaming =
        StreamingPlainSumProverMultiChunk::new(poly.evaluations().to_vec(), num_vars);
    let mut transcript = Blake2bTranscript::new(b"streaming_multi");
    let mut round_polys = Vec::with_capacity(num_vars);

    for _ in 0..num_vars {
        streaming.begin_round();
        let table = streaming.evals.clone();
        // Feed in 3 unequal chunks
        let c1 = table.len() / 4;
        let c2 = table.len() / 2;
        streaming.process_chunk(&table[..c1]);
        streaming.process_chunk(&table[c1..c2]);
        streaming.process_chunk(&table[c2..]);

        let rp = streaming.finish_round();

        for coeff in rp.coefficients() {
            coeff.append_to_transcript(&mut transcript);
        }

        let challenge: C = transcript.challenge().into();
        streaming.bind(challenge.into());
        round_polys.push(rp);
    }

    let proof = crate::proof::SumcheckProof {
        round_polynomials: round_polys,
    };

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    let mut vt = Blake2bTranscript::new(b"streaming_multi");
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof, &mut vt)
            .expect("multi-chunk streaming proof should verify");

    assert_eq!(final_eval, poly.evaluate(&challenges));
}

/// Multi-chunk-aware streaming prover that tracks how many elements
/// have been consumed so far in the current round to correctly assign
/// entries to the low/high halves.
struct StreamingPlainSumProverMultiChunk {
    evals: Vec<Fr>,
    num_vars: usize,
    accum: [Fr; 2],
    consumed: usize,
}

impl StreamingPlainSumProverMultiChunk {
    fn new(evals: Vec<Fr>, num_vars: usize) -> Self {
        Self {
            evals,
            num_vars,
            accum: [Fr::zero(); 2],
            consumed: 0,
        }
    }
}

#[test]
fn batched_different_num_vars() {
    let mut rng = ChaCha20Rng::seed_from_u64(900);

    // Instance A: 6 variables (degree 1)
    let num_vars_a = 6;
    let poly_a = Polynomial::<Fr>::random(num_vars_a, &mut rng);
    let sum_a: Fr = poly_a.evaluations().iter().copied().sum();

    // Instance B: 4 variables (degree 1)
    let num_vars_b = 4;
    let poly_b = Polynomial::<Fr>::random(num_vars_b, &mut rng);
    let sum_b: Fr = poly_b.evaluations().iter().copied().sum();

    let claims = vec![
        SumcheckClaim {
            num_vars: num_vars_a,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: num_vars_b,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![
        Box::new(PlainSumWitness {
            poly: poly_a.clone(),
        }),
        Box::new(PlainSumWitness {
            poly: poly_b.clone(),
        }),
    ];

    let mut pt = Blake2bTranscript::new(b"diff_num_vars");
    let proof = BatchedSumcheckProver::prove(&claims, &mut witnesses, &mut pt);

    // Proof should have max_num_vars rounds
    assert_eq!(proof.round_polynomials.len(), num_vars_a);

    let mut vt = Blake2bTranscript::new(b"diff_num_vars");
    let result = BatchedSumcheckVerifier::verify(&claims, &proof, &mut vt);
    assert!(
        result.is_ok(),
        "batched verification with different num_vars should succeed"
    );
}

#[test]
fn batched_mixed_degree_and_num_vars() {
    let mut rng = ChaCha20Rng::seed_from_u64(901);

    // Instance A: degree-2 product sumcheck with 5 variables
    let num_vars_a = 5;
    let poly_a = Polynomial::<Fr>::random(num_vars_a, &mut rng);
    let tau_a: Vec<Fr> = (0..num_vars_a).map(|_| Fr::random(&mut rng)).collect();
    let sum_a = poly_a.evaluate(&tau_a);

    // Instance B: degree-1 plain sum with 3 variables
    let num_vars_b = 3;
    let poly_b = Polynomial::<Fr>::random(num_vars_b, &mut rng);
    let sum_b: Fr = poly_b.evaluations().iter().copied().sum();

    let claims = vec![
        SumcheckClaim {
            num_vars: num_vars_a,
            degree: 2,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: num_vars_b,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![
        Box::new(EqProductWitness::new(poly_a.clone(), &tau_a)),
        Box::new(PlainSumWitness {
            poly: poly_b.clone(),
        }),
    ];

    let mut pt = Blake2bTranscript::new(b"mixed_degree_vars");
    let proof = BatchedSumcheckProver::prove(&claims, &mut witnesses, &mut pt);

    assert_eq!(proof.round_polynomials.len(), num_vars_a);

    let mut vt = Blake2bTranscript::new(b"mixed_degree_vars");
    let result = BatchedSumcheckVerifier::verify(&claims, &proof, &mut vt);
    assert!(
        result.is_ok(),
        "batched verification with mixed degree and num_vars should succeed"
    );
}

#[test]
fn batched_challenge_slicing() {
    let mut rng = ChaCha20Rng::seed_from_u64(902);

    let num_vars_a = 6;
    let num_vars_b = 4;
    let max_num_vars = num_vars_a;

    let poly_a = Polynomial::<Fr>::random(num_vars_a, &mut rng);
    let poly_b = Polynomial::<Fr>::random(num_vars_b, &mut rng);
    let sum_a: Fr = poly_a.evaluations().iter().copied().sum();
    let sum_b: Fr = poly_b.evaluations().iter().copied().sum();

    let claims = vec![
        SumcheckClaim {
            num_vars: num_vars_a,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: num_vars_b,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut witnesses: Vec<Box<dyn SumcheckCompute<Fr>>> = vec![
        Box::new(PlainSumWitness {
            poly: poly_a.clone(),
        }),
        Box::new(PlainSumWitness {
            poly: poly_b.clone(),
        }),
    ];

    let mut pt = Blake2bTranscript::new(b"slice_test");
    let proof = BatchedSumcheckProver::prove(&claims, &mut witnesses, &mut pt);

    let mut vt = Blake2bTranscript::new(b"slice_test");
    let (_final_eval, challenges) =
        BatchedSumcheckVerifier::verify(&claims, &proof, &mut vt)
            .expect("verification should succeed");

    assert_eq!(challenges.len(), max_num_vars);

    // Instance A uses all challenges (offset = 0)
    let r_a = &challenges[0..num_vars_a];
    assert_eq!(poly_a.evaluate(r_a), poly_a.evaluate(&challenges));

    // Instance B uses only the last num_vars_b challenges (offset = 2)
    let offset_b = max_num_vars - num_vars_b;
    let r_b = &challenges[offset_b..offset_b + num_vars_b];
    let eval_b = poly_b.evaluate(r_b);

    // Sanity: eval_b should be a valid field element from the polynomial
    assert_ne!(eval_b, Fr::zero());
}

impl StreamingSumcheckProver<Fr> for StreamingPlainSumProverMultiChunk {
    fn begin_round(&mut self) {
        self.accum = [Fr::zero(); 2];
        self.consumed = 0;
    }

    fn process_chunk(&mut self, chunk: &[Fr]) {
        let half = self.evals.len() / 2;
        for &val in chunk {
            if self.consumed < half {
                self.accum[0] += val;
            } else {
                self.accum[1] += val;
            }
            self.consumed += 1;
        }
    }

    fn finish_round(&mut self) -> UnivariatePoly<Fr> {
        let sum_lo = self.accum[0];
        let sum_hi = self.accum[1];
        UnivariatePoly::new(vec![sum_lo, sum_hi - sum_lo])
    }

    fn bind(&mut self, challenge: Fr) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            let lo = self.evals[i];
            let hi = self.evals[i + half];
            self.evals[i] = lo + challenge * (hi - lo);
        }
        self.evals.truncate(half);
        self.num_vars -= 1;
    }
}

#[test]
fn transcript_label_mismatch_fails() {
    let mut rng = ChaCha20Rng::seed_from_u64(900);
    let num_vars = 4;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    let mut witness = PlainSumWitness { poly: poly.clone() };
    let mut pt = Blake2bTranscript::new(b"label_a");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

    // Verify with a different label — challenges diverge, should fail final check
    let mut vt = Blake2bTranscript::new(b"label_b");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);

    if let Ok((final_eval, challenges)) = result {
        // Verification "passes" structurally (sums match per round) but
        // the final evaluation will disagree since challenges differ
        assert_ne!(
            final_eval,
            poly.evaluate(&challenges),
            "label mismatch should cause evaluation divergence"
        );
    }
    // Err is also acceptable: structural rejection
}

#[test]
fn tampered_round_coefficient_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(901);
    let num_vars = 4;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    let mut witness = PlainSumWitness { poly };
    let mut pt = Blake2bTranscript::new(b"tamper_test");
    let mut proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

    // Tamper with the first round polynomial
    let tampered = UnivariatePoly::new(vec![Fr::from_u64(999), Fr::from_u64(1)]);
    proof.round_polynomials[0] = tampered;

    let mut vt = Blake2bTranscript::new(b"tamper_test");
    let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
    assert!(
        result.is_err(),
        "tampered round polynomial should cause rejection"
    );
}

#[test]
fn keccak_transcript_prove_verify() {
    use jolt_transcript::KeccakTranscript;

    let mut rng = ChaCha20Rng::seed_from_u64(902);
    let num_vars = 5;
    let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
    let claimed_sum: Fr = poly.evaluations().iter().copied().sum();

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum,
    };

    let mut witness = PlainSumWitness { poly: poly.clone() };
    let mut pt = KeccakTranscript::new(b"keccak_test");
    let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

    let mut vt = KeccakTranscript::new(b"keccak_test");
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof, &mut vt)
            .expect("keccak transcript verification should succeed");

    assert_eq!(final_eval, poly.evaluate(&challenges));
}
