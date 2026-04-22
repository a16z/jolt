//! Unit tests for sumcheck verification.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests may panic on assertion failures"
)]

use jolt_field::{Field, Fr};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::proof::SumcheckProof;
use crate::round::{ClearRoundVerifier, RoundVerifier};
use crate::verifier::SumcheckVerifier;
use crate::BatchedSumcheckVerifier;

type F = Fr;

/// Build an honest sumcheck proof for a multilinear polynomial given
/// as evaluations over {0,1}^n.
///
/// This is a minimal reference prover: in each round it computes the
/// round polynomial by partial evaluation, absorbs it into the
/// transcript, squeezes a challenge, and binds.
fn honest_prove(
    evals: &[F],
    num_vars: usize,
    transcript: &mut Blake2bTranscript,
) -> SumcheckProof<F> {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = buf.len() / 2;

        // HighToLow binding: pairs (buf[i], buf[i + half])
        // s(X) = sum_i [ buf[i] * (1-X) + buf[i + half] * X ]
        let mut eval_0 = F::from_u64(0);
        let mut eval_1 = F::from_u64(0);
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }

        // Degree-1 round polynomial: s(X) = eval_0 + (eval_1 - eval_0) * X
        let c0 = eval_0;
        let c1 = eval_1 - eval_0;
        let round_poly = UnivariatePoly::new(vec![c0, c1]);

        // Absorb (matching ClearRoundVerifier::new() — no label)
        for &coeff in round_poly.coefficients() {
            coeff.append_to_transcript(transcript);
        }

        let r: F = transcript.challenge();
        round_polys.push(round_poly);

        // Bind HighToLow: buf[i] = buf[i] + r * (buf[i + half] - buf[i])
        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    SumcheckProof {
        round_polynomials: round_polys,
    }
}

/// Compute the sum over {0,1}^n of multilinear evaluations.
fn compute_sum(evals: &[F]) -> F {
    evals.iter().copied().sum()
}

#[test]
fn verify_valid_degree1_proof() {
    // f(x1, x2, x3) with known evaluations
    // f(0,0,0)=1, f(1,0,0)=2, f(0,1,0)=3, f(1,1,0)=4,
    // f(0,0,1)=5, f(1,0,1)=6, f(0,1,1)=7, f(1,1,1)=8
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);
    let num_vars = 3;

    let mut prover_transcript = Blake2bTranscript::new(b"sumcheck-test");
    let proof = honest_prove(&evals, num_vars, &mut prover_transcript);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut verifier_transcript = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &mut verifier_transcript,
        &clear,
    );
    assert!(result.is_ok(), "verification failed: {:?}", result.err());

    let (final_eval, challenges) = result.unwrap();
    assert_eq!(challenges.len(), num_vars);

    // Verify the final evaluation matches direct evaluation at the challenge point.
    // evaluate_and_consume binds variables in order, matching the sumcheck bind order.
    let poly = jolt_poly::Polynomial::new(evals);
    let expected = poly.evaluate_and_consume(&challenges);
    assert_eq!(final_eval, expected);
}

#[test]
fn verify_single_variable() {
    // f(x) = 3 + 7x, evals = [3, 10]
    let evals = vec![F::from_u64(3), F::from_u64(10)];
    let sum = compute_sum(&evals); // 13

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let proof = honest_prove(&evals, 1, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let (final_eval, challenges) =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear).unwrap();
    assert_eq!(challenges.len(), 1);

    let poly = jolt_poly::Polynomial::new(evals);
    assert_eq!(final_eval, poly.evaluate_and_consume(&challenges));
}

#[test]
fn verify_round_check_failure() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Corrupt the first round polynomial
    let bad_coeffs = vec![F::from_u64(999), F::from_u64(1)];
    proof.round_polynomials[0] = UnivariatePoly::new(bad_coeffs);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_err());
    match result.unwrap_err() {
        SumcheckError::RoundCheckFailed { round, .. } => assert_eq!(round, 0),
        other => panic!("expected RoundCheckFailed, got {other:?}"),
    }
}

#[test]
fn verify_wrong_num_rounds() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Remove the last round
    let _ = proof.round_polynomials.pop();

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    match result.unwrap_err() {
        SumcheckError::WrongNumberOfRounds { expected, got } => {
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        other => panic!("expected WrongNumberOfRounds, got {other:?}"),
    }
}

#[test]
fn verify_degree_exceeded() {
    let evals: Vec<F> = (1..=4).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let mut proof = honest_prove(&evals, 2, &mut pt);

    // Replace first round poly with a degree-3 polynomial (4 coefficients)
    proof.round_polynomials[0] =
        UnivariatePoly::new(vec![sum, F::from_u64(1), F::from_u64(0), F::from_u64(1)]);

    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1, // degree bound is 1, but we gave degree 3
        claimed_sum: sum,
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    match result.unwrap_err() {
        SumcheckError::DegreeBoundExceeded { got, max } => {
            assert_eq!(max, 1);
            assert!(got > 1);
        }
        other => panic!("expected DegreeBoundExceeded, got {other:?}"),
    }
}

#[test]
fn verify_wrong_claimed_sum() {
    let evals: Vec<F> = (1..=4).map(F::from_u64).collect();
    let real_sum = compute_sum(&evals);

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let proof = honest_prove(&evals, 2, &mut pt);

    // Claim a different sum
    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: real_sum + F::from_u64(1),
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        SumcheckError::RoundCheckFailed { round: 0, .. }
    ));
}

#[test]
fn clear_round_verifier_with_label_absorbs_label() {
    // poly(0) = 5, poly(1) = 5 + 3 = 8, sum = 13
    let poly = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);
    let running_sum = F::from_u64(13);
    let label = b"test_label";
    let round_verifier = ClearRoundVerifier::with_label(label);

    let mut t1 = Blake2bTranscript::new(b"sumcheck-test");

    round_verifier
        .absorb_and_check(&poly, running_sum, 1, 0, &mut t1)
        .unwrap();
    let c1: F = t1.challenge();

    // Absorb manually (should match)
    let mut t2 = Blake2bTranscript::new(b"sumcheck-test");
    t2.append(&LabelWithCount(label, 2));
    for coeff in poly.coefficients() {
        coeff.append_to_transcript(&mut t2);
    }
    let c2: F = t2.challenge();

    assert_eq!(c1, c2, "labeled absorption must match manual absorption");
}

#[test]
fn clear_round_verifier_no_label() {
    let poly = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);
    let running_sum = F::from_u64(13); // 5 + 8

    let round_verifier = ClearRoundVerifier::new();

    let mut t1 = Blake2bTranscript::new(b"sumcheck-test");

    round_verifier
        .absorb_and_check(&poly, running_sum, 1, 0, &mut t1)
        .unwrap();
    let c1: F = t1.challenge();

    // Manual: just coefficients, no label
    let mut t2 = Blake2bTranscript::new(b"sumcheck-test");
    for coeff in poly.coefficients() {
        coeff.append_to_transcript(&mut t2);
    }
    let c2: F = t2.challenge();

    assert_eq!(c1, c2, "unlabeled absorption must match manual absorption");
}

#[test]
fn clear_round_verifier_compressed_matches_manual_absorption() {
    // s(X) = 2 + 3*X + 5*X^2  ⇒  s(0) = 2, s(1) = 10, running_sum = 12.
    // Sanity: 2*c0 + c1 + c2 = 2*2 + 3 + 5 = 12.
    let poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let running_sum = F::from_u64(12);
    let label = b"compressed_test";
    let round_verifier = ClearRoundVerifier::with_label_compressed(label);

    let mut t1 = Blake2bTranscript::new(b"sumcheck-test");
    round_verifier
        .absorb_and_check(&poly, running_sum, 2, 0, &mut t1)
        .unwrap();
    let ch1: F = t1.challenge();

    // Manual absorb matching the compressed wire format: label_with_count(d), c0, c2..cd.
    let mut t2 = Blake2bTranscript::new(b"sumcheck-test");
    let coeffs = poly.coefficients();
    t2.append(&LabelWithCount(label, (coeffs.len() - 1) as u64));
    coeffs[0].append_to_transcript(&mut t2);
    for c in coeffs.iter().skip(2) {
        c.append_to_transcript(&mut t2);
    }
    let ch2: F = t2.challenge();

    assert_eq!(
        ch1, ch2,
        "compressed absorption must omit c1 and match manual c0+c2..cd absorption"
    );
}

#[test]
fn clear_round_verifier_compressed_rejects_wrong_running_sum() {
    // Even when the linear term is omitted from absorption, the sum check
    // (s(0) + s(1) == running_sum) still binds every coefficient.
    let poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let wrong_running_sum = F::from_u64(999);
    let round_verifier = ClearRoundVerifier::with_label_compressed(b"compressed_test");

    let mut t = Blake2bTranscript::<F>::new(b"sumcheck-test");
    let result = round_verifier.absorb_and_check(&poly, wrong_running_sum, 2, 0, &mut t);
    assert!(
        matches!(
            result,
            Err(SumcheckError::RoundCheckFailed { round: 0, .. })
        ),
        "compressed verifier must enforce s(0)+s(1)==running_sum: {result:?}"
    );
}

#[test]
fn clear_round_verifier_compressed_rejects_short_polynomial() {
    // Compressed encoding omits the linear term; a polynomial with fewer than
    // two coefficients has no linear term to recover and is malformed.
    let degree_zero = UnivariatePoly::new(vec![F::from_u64(7)]);
    let round_verifier = ClearRoundVerifier::with_label_compressed(b"compressed_test");

    let mut t = Blake2bTranscript::<F>::new(b"sumcheck-test");
    let result = round_verifier.absorb_and_check(&degree_zero, F::from_u64(14), 2, 3, &mut t);
    assert!(
        matches!(
            result,
            Err(SumcheckError::CompressedPolynomialTooShort { round: 3, got: 1 })
        ),
        "compressed verifier must reject degree-0 polynomials: {result:?}"
    );
}

#[test]
fn batched_verify_same_size() {
    // Two polynomials, both 2 variables
    let evals_a: Vec<F> = (1..=4).map(F::from_u64).collect();
    let evals_b: Vec<F> = (5..=8).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

    // Prove: absorb claims, squeeze alpha, combine, prove combined
    let mut pt = Blake2bTranscript::new(b"sumcheck-test");

    sum_a.append_to_transcript(&mut pt);
    sum_b.append_to_transcript(&mut pt);
    let alpha: F = pt.challenge();

    // Combined polynomial: evals_a[i] + alpha * evals_b[i]
    let combined: Vec<F> = evals_a
        .iter()
        .zip(&evals_b)
        .map(|(&a, &b)| a + alpha * b)
        .collect();
    let proof = honest_prove(&combined, 2, &mut pt);

    let claims = vec![
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result =
        BatchedSumcheckVerifier::verify(&claims, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_ok(), "batched verify failed: {:?}", result.err());

    let (_final_eval, challenges) = result.unwrap();
    assert_eq!(challenges.len(), 2);
}

#[test]
fn batched_verify_different_sizes() {
    // Claim A: 3 variables, Claim B: 2 variables
    let evals_a: Vec<F> = (1..=8).map(F::from_u64).collect();
    let evals_b: Vec<F> = (1..=4).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

    let max_vars = 3;

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");

    sum_a.append_to_transcript(&mut pt);
    sum_b.append_to_transcript(&mut pt);
    let alpha: F = pt.challenge();

    // B is scaled by 2^(3-2) = 2 for front-loaded padding.
    // Combined over 2^3 = 8 points:
    // For the first round (the padding round for B), B contributes a constant
    // sum_b_scaled / 2 to both s(0) and s(1).
    //
    // To build the combined polynomial correctly:
    // A has 8 evals, B has 4 evals extended to 8 by duplicating (each eval appears twice,
    // since the first variable doesn't affect B).
    let evals_b_extended: Vec<F> = evals_b.iter().flat_map(|&v| [v, v]).collect();

    let combined: Vec<F> = evals_a
        .iter()
        .zip(&evals_b_extended)
        .map(|(&a, &b)| a + alpha * b)
        .collect();
    let proof = honest_prove(&combined, max_vars, &mut pt);

    let claims = vec![
        SumcheckClaim {
            num_vars: 3,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result =
        BatchedSumcheckVerifier::verify(&claims, &proof.round_polynomials, &mut vt, &clear);
    assert!(result.is_ok(), "batched verify failed: {:?}", result.err());
}

#[test]
fn batched_single_claim_matches_single_verify() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    // The batched verifier absorbs the claim and squeezes alpha even for a
    // single claim, so the transcript diverges from the single verifier.
    // But internally it should still produce a valid verification.
    let mut pt = Blake2bTranscript::new(b"sumcheck-test");

    sum.append_to_transcript(&mut pt);
    let _alpha: F = pt.challenge();

    // alpha^0 = 1, so combined polynomial = evals (single claim)
    let proof = honest_prove(&evals, 3, &mut pt);

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result =
        BatchedSumcheckVerifier::verify(&[claim], &proof.round_polynomials, &mut vt, &clear);
    assert!(
        result.is_ok(),
        "single-claim batch failed: {:?}",
        result.err()
    );
}

#[test]
fn batched_empty_claims_returns_error() {
    let claims: &[SumcheckClaim<F>] = &[];
    let round_proofs: &[UnivariatePoly<F>] = &[];
    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let clear = ClearRoundVerifier::new();
    let result = BatchedSumcheckVerifier::verify(claims, round_proofs, &mut vt, &clear);
    assert!(matches!(result, Err(SumcheckError::EmptyClaims)));
}

/// A mock round verifier that always accepts and returns a fixed running sum.
/// Used to verify that `SumcheckVerifier` dispatches through the trait correctly.
struct MockRoundVerifier {
    fixed_sum: F,
}

impl crate::round::RoundVerifier<F> for MockRoundVerifier {
    type RoundProof = ();

    fn absorb_and_check(
        &self,
        _proof: &(),
        _running_sum: F,
        _degree_bound: usize,
        _round: usize,
        transcript: &mut impl jolt_transcript::Transcript,
    ) -> Result<(), SumcheckError> {
        // Absorb a deterministic byte so challenges are well-defined
        F::from_u64(42).append_to_transcript(transcript);
        Ok(())
    }

    fn next_running_sum(&self, _proof: &(), _challenge: F) -> F {
        self.fixed_sum
    }
}

#[test]
fn verify_dispatches_through_round_verifier_trait() {
    let fixed = F::from_u64(99);
    let mock = MockRoundVerifier { fixed_sum: fixed };

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: F::from_u64(0), // doesn't matter — mock always accepts
    };
    let round_proofs = [(), (), ()];

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let result = SumcheckVerifier::verify(&claim, &round_proofs, &mut vt, &mock);
    assert!(
        result.is_ok(),
        "mock verifier should accept: {:?}",
        result.err()
    );

    let (final_eval, challenges) = result.unwrap();
    assert_eq!(challenges.len(), 3);
    // Mock always returns fixed_sum, so final eval should be fixed
    assert_eq!(final_eval, fixed);
}
