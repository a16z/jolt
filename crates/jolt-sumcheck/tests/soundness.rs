//! Soundness tests: adversarial scenarios for sumcheck verification.
//!
//! These tests probe whether a malicious prover can trick the verifier into
//! accepting an invalid claim. Each test targets a specific attack vector
//! against the sumcheck protocol.

#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use jolt_field::{Field, Fr};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::{EvaluationClaim, SumcheckClaim};
use jolt_sumcheck::error::SumcheckError;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_sumcheck::round_proof::RoundProof;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};

type F = Fr;

#[derive(Debug)]
enum OracleCheckError {
    #[expect(
        dead_code,
        reason = "inner error shown via Debug in test panic messages"
    )]
    Sumcheck(SumcheckError<F>),
    FinalEvalMismatch,
}

impl From<SumcheckError<F>> for OracleCheckError {
    fn from(err: SumcheckError<F>) -> Self {
        OracleCheckError::Sumcheck(err)
    }
}

fn new_transcript() -> Blake2bTranscript {
    Blake2bTranscript::new(b"soundness-test")
}

/// Honest degree-1 sumcheck prover.
fn honest_prove(
    evals: &[F],
    num_vars: usize,
    transcript: &mut Blake2bTranscript,
) -> SumcheckProof<F> {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = buf.len() / 2;
        let mut eval_0 = F::from_u64(0);
        let mut eval_1 = F::from_u64(0);
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }
        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);
        <UnivariatePoly<F> as RoundProof<F>>::append_to_transcript(&round_poly, transcript);
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

fn compute_sum(evals: &[F]) -> F {
    evals.iter().copied().sum()
}

/// Full verification pipeline: sumcheck round checks + oracle evaluation check.
///
/// Returns the challenge vector on success. Returns
/// `OracleCheckError::FinalEvalMismatch` if the proof passes all round checks
/// but the final evaluation doesn't match the intended polynomial.
fn verify_with_oracle_check(
    claim: &SumcheckClaim<F>,
    proof: &SumcheckProof<F>,
    intended_evals: &[F],
) -> Result<Vec<F>, OracleCheckError> {
    let mut transcript = new_transcript();
    let EvaluationClaim {
        point: challenges,
        value: final_eval,
    } = SumcheckVerifier::verify(claim, &proof.round_polynomials, &mut transcript)?;

    let expected = Polynomial::new(intended_evals.to_vec()).evaluate_and_consume(&challenges);
    if final_eval != expected {
        return Err(OracleCheckError::FinalEvalMismatch);
    }

    Ok(challenges)
}

#[test]
fn honest_proof_passes_oracle_check() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let proof = honest_prove(&evals, 3, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let result = verify_with_oracle_check(&claim, &proof, &evals);
    assert!(result.is_ok());
}

#[test]
fn wrong_polynomial_same_sum_fails_oracle_check() {
    // f and g have the same sum but are different polynomials.
    // An honest proof for g will pass all round checks when verified against
    // claimed_sum = sum(g), but the final evaluation will correspond to g(r),
    // not f(r). The oracle check catches this.
    let f_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let g_evals: Vec<F> = (1..=8).rev().map(F::from_u64).collect();

    let sum_f = compute_sum(&f_evals);
    let sum_g = compute_sum(&g_evals);
    assert_eq!(sum_f, sum_g, "precondition: f and g must have equal sums");

    // Construct honest proof for g
    let mut pt = new_transcript();
    let proof = honest_prove(&g_evals, 3, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum_g,
    };

    // Round checks pass (proof is internally consistent for g)
    let mut vt = new_transcript();
    let round_result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);
    assert!(
        round_result.is_ok(),
        "round checks should pass for honest g proof"
    );

    // But oracle check against f fails — the final eval is g(r), not f(r)
    let result = verify_with_oracle_check(&claim, &proof, &f_evals);
    assert!(
        matches!(result, Err(OracleCheckError::FinalEvalMismatch)),
        "oracle check must catch polynomial substitution: {:?}",
        result
    );
}

#[test]
fn proof_for_different_polynomial_different_sum_fails_round_check() {
    // Construct an honest proof for g, but claim it proves f (different sum).
    // The round check at round 0 must fail because the proof's s_0(0) + s_0(1)
    // was computed for sum(g), not sum(f).
    let f_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let g_evals: Vec<F> = (10..=17).map(F::from_u64).collect();

    let sum_f = compute_sum(&f_evals);
    let sum_g = compute_sum(&g_evals);
    assert_ne!(sum_f, sum_g);

    let mut pt = new_transcript();
    let proof = honest_prove(&g_evals, 3, &mut pt);

    // Claim sum(f) but provide proof for g
    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum_f,
    };

    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);
    assert!(matches!(
        result,
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));
}

#[test]
fn corrupted_middle_round_detected() {
    let evals: Vec<F> = (1..=16).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let mut proof = honest_prove(&evals, 4, &mut pt);

    // Corrupt round 2 (middle round): replace with arbitrary polynomial
    // that has the same degree but wrong s(0) + s(1).
    proof.round_polynomials[2] = UnivariatePoly::new(vec![F::from_u64(999), F::from_u64(1)]);

    let claim = SumcheckClaim {
        num_vars: 4,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);

    // Corruption at round 2 may be detected at round 2 (wrong sum) or later
    // (transcript desync from corrupted absorption). Either way, it must fail.
    assert!(result.is_err(), "corrupted middle round must be rejected");
}

#[test]
fn corrupted_last_round_detected() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Corrupt only the last round polynomial
    proof.round_polynomials[2] = UnivariatePoly::new(vec![F::from_u64(0), F::from_u64(0)]);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);
    assert!(result.is_err(), "corrupted last round must be rejected");
}

#[test]
fn swapped_round_order_rejected() {
    // Swap rounds 0 and 1. The proof is no longer internally consistent because
    // each round's polynomial depends on the previous challenge, and swapping
    // changes the Fiat-Shamir transcript.
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let mut proof = honest_prove(&evals, 3, &mut pt);

    proof.round_polynomials.swap(0, 1);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);

    // Round 0 now has the wrong s(0)+s(1) (it was computed for a different running sum).
    // Even if by accident s(0)+s(1) matched, the transcript would desync.
    assert!(result.is_err(), "swapped round order must be rejected");
}

#[test]
fn replayed_round_polynomial_rejected() {
    // Use the first round's polynomial for every round. The running sum check
    // will fail because the replayed polynomial doesn't satisfy s(0)+s(1) == running_sum
    // for rounds > 0.
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let proof = honest_prove(&evals, 3, &mut pt);

    let replayed = SumcheckProof {
        round_polynomials: vec![proof.round_polynomials[0].clone(); 3],
    };

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, &replayed.round_polynomials, &mut vt);
    assert!(result.is_err(), "replayed rounds must be rejected");
}

#[test]
fn all_zero_round_polynomials_rejected_for_nonzero_sum() {
    // If the sum is nonzero, round 0 requires s(0) + s(1) == sum.
    // All-zero polynomials have s(0) + s(1) = 0, so this must fail.
    let evals: Vec<F> = (1..=4).map(F::from_u64).collect();
    let sum = compute_sum(&evals);
    assert_ne!(sum, F::from_u64(0));

    let zero_poly = UnivariatePoly::new(vec![F::from_u64(0)]);
    let proof = SumcheckProof {
        round_polynomials: vec![zero_poly; 2],
    };

    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);
    assert!(matches!(
        result,
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));
}

#[test]
fn all_zero_polynomial_honest_proof_for_zero_sum() {
    // The zero polynomial f(x) = 0 for all x has sum = 0.
    // An honest proof should be all-zero round polynomials and must verify.
    let num_vars = 3;
    let evals = vec![F::from_u64(0); 1 << num_vars];

    let mut pt = new_transcript();
    let proof = honest_prove(&evals, num_vars, &mut pt);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let result = verify_with_oracle_check(&claim, &proof, &evals);
    assert!(
        result.is_ok(),
        "zero polynomial must verify: {:?}",
        result.err()
    );
}

#[test]
fn verifier_transcript_desync_rejected() {
    // If the verifier's transcript has been poisoned with extra data before
    // verification begins, the Fiat-Shamir challenges will differ from the
    // prover's, causing the running sum check to fail.
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let proof = honest_prove(&evals, 3, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    // Poison the verifier transcript with extra data
    let mut vt = new_transcript();
    F::from_u64(0xdead).append_to_transcript(&mut vt);

    let result = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt);

    // Round 0's s(0)+s(1) check passes (it doesn't depend on challenges),
    // but the challenge r_0 will differ, so round 1's running sum will be wrong.
    // For num_vars == 1 this wouldn't be caught by round checks (only by oracle check),
    // but for num_vars >= 2, the transcript desync propagates to a round check failure.
    assert!(result.is_err(), "transcript desync must be rejected");
}

#[test]
fn num_vars_zero_accepts_any_claimed_sum() {
    // With 0 variables, the "polynomial" is a constant. The sum over the empty
    // hypercube {0,1}^0 = {()} is just the constant value itself.
    // The verifier should accept with no rounds and return (claimed_sum, []).
    let claim = SumcheckClaim {
        num_vars: 0,
        degree: 1,
        claimed_sum: F::from_u64(42),
    };

    let round_proofs: &[UnivariatePoly<F>] = &[];
    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, round_proofs, &mut vt);
    assert!(result.is_ok());

    let EvaluationClaim {
        point: challenges,
        value: final_eval,
    } = result.unwrap();
    assert_eq!(final_eval, F::from_u64(42));
    assert!(challenges.is_empty());
}

#[test]
fn num_vars_zero_no_oracle_check_possible() {
    // With 0 rounds, the verifier returns the claimed_sum as-is with no
    // Fiat-Shamir interaction. The protocol offers no soundness guarantee
    // at this point — security relies entirely on the oracle check.
    // A malicious prover can claim any sum and "verify" it.
    let claim = SumcheckClaim {
        num_vars: 0,
        degree: 1,
        claimed_sum: F::from_u64(999), // arbitrary lie
    };

    let round_proofs: &[UnivariatePoly<F>] = &[];
    let mut vt = new_transcript();
    let result = SumcheckVerifier::verify(&claim, round_proofs, &mut vt);

    // Passes — the verifier has nothing to check!
    // Only the oracle check (comparing 999 against the actual constant) catches this.
    assert!(result.is_ok());
    assert_eq!(result.unwrap().value, F::from_u64(999));
}

#[test]
fn constant_polynomial_all_same_evals() {
    // f(x) = 7 for all x in {0,1}^3 → sum = 7 * 8 = 56
    let num_vars = 3;
    let evals = vec![F::from_u64(7); 1 << num_vars];
    let sum = compute_sum(&evals);

    let mut pt = new_transcript();
    let proof = honest_prove(&evals, num_vars, &mut pt);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let result = verify_with_oracle_check(&claim, &proof, &evals);
    assert!(result.is_ok());

    // The final eval should be 7 regardless of the challenge point,
    // since f is constant.
    let mut vt = new_transcript();
    let final_eval = SumcheckVerifier::verify(&claim, &proof.round_polynomials, &mut vt)
        .unwrap()
        .value;
    assert_eq!(final_eval, F::from_u64(7));
}

#[test]
fn batched_one_dishonest_claim_rejected() {
    // Batch two claims: one honest, one with a wrong sum.
    // The combined proof must fail because the combined sum won't match.
    let evals_a: Vec<F> = (1..=4).map(F::from_u64).collect();
    let evals_b: Vec<F> = (5..=8).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

    // Build combined proof honestly
    let mut pt = new_transcript();
    sum_a.append_to_transcript(&mut pt);
    sum_b.append_to_transcript(&mut pt);
    let alpha: F = pt.challenge();

    let combined: Vec<F> = evals_a
        .iter()
        .zip(&evals_b)
        .map(|(&a, &b)| a + alpha * b)
        .collect();
    let proof = honest_prove(&combined, 2, &mut pt);

    // Claim B lies about its sum
    let claims = vec![
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_b + F::from_u64(1), // lie
        },
    ];

    let mut vt = new_transcript();
    let result = BatchedSumcheckVerifier::verify(&claims, &proof.round_polynomials, &mut vt);
    assert!(result.is_err(), "dishonest claim in batch must be rejected");
}

#[test]
fn batched_swapped_claim_order_rejected() {
    // If claims are presented in a different order than what the prover used,
    // the batching coefficients alpha^j are applied to the wrong claims,
    // producing a different combined sum.
    let evals_a: Vec<F> = (1..=4).map(F::from_u64).collect();
    let evals_b: Vec<F> = (5..=8).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

    // Prove with order [a, b]
    let mut pt = new_transcript();
    sum_a.append_to_transcript(&mut pt);
    sum_b.append_to_transcript(&mut pt);
    let alpha: F = pt.challenge();

    let combined: Vec<F> = evals_a
        .iter()
        .zip(&evals_b)
        .map(|(&a, &b)| a + alpha * b)
        .collect();
    let proof = honest_prove(&combined, 2, &mut pt);

    // Verify with order [b, a] — swapped
    let claims_swapped = vec![
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_b,
        },
        SumcheckClaim {
            num_vars: 2,
            degree: 1,
            claimed_sum: sum_a,
        },
    ];

    let mut vt = new_transcript();
    let result =
        BatchedSumcheckVerifier::verify(&claims_swapped, &proof.round_polynomials, &mut vt);
    assert!(result.is_err(), "swapped claim order must be rejected");
}
