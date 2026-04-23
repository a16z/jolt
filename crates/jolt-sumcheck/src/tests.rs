//! Unit tests for sumcheck verification.

use jolt_field::{Field, Fr};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{AppendToTranscript, LabelWithCount, MockTranscript, Transcript};
use jolt_verifier_backend::FieldBackend;

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
fn honest_prove<T>(evals: &[F], num_vars: usize, transcript: &mut T) -> SumcheckProof<F>
where
    T: Transcript<Challenge = F>,
{
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

    let mut prover_transcript = MockTranscript::<F>::default();
    let proof = honest_prove(&evals, num_vars, &mut prover_transcript);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut verifier_transcript = MockTranscript::<F>::default();
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

    let mut pt = MockTranscript::<F>::default();
    let proof = honest_prove(&evals, 1, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = MockTranscript::<F>::default();
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

    let mut pt = MockTranscript::<F>::default();
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Corrupt the first round polynomial
    let bad_coeffs = vec![F::from_u64(999), F::from_u64(1)];
    proof.round_polynomials[0] = UnivariatePoly::new(bad_coeffs);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = MockTranscript::<F>::default();
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

    let mut pt = MockTranscript::<F>::default();
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Remove the last round
    let _ = proof.round_polynomials.pop();

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = MockTranscript::<F>::default();
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

    let mut pt = MockTranscript::<F>::default();
    let mut proof = honest_prove(&evals, 2, &mut pt);

    // Replace first round poly with a degree-3 polynomial (4 coefficients)
    proof.round_polynomials[0] =
        UnivariatePoly::new(vec![sum, F::from_u64(1), F::from_u64(0), F::from_u64(1)]);

    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1, // degree bound is 1, but we gave degree 3
        claimed_sum: sum,
    };

    let mut vt = MockTranscript::<F>::default();
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

    let mut pt = MockTranscript::<F>::default();
    let proof = honest_prove(&evals, 2, &mut pt);

    // Claim a different sum
    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: real_sum + F::from_u64(1),
    };

    let mut vt = MockTranscript::<F>::default();
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

    let mut t1 = MockTranscript::<F>::default();

    round_verifier
        .absorb_and_check(&poly, running_sum, 1, 0, &mut t1)
        .unwrap();
    let c1: F = t1.challenge();

    // Absorb manually (should match)
    let mut t2 = MockTranscript::<F>::default();
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

    let mut t1 = MockTranscript::<F>::default();

    round_verifier
        .absorb_and_check(&poly, running_sum, 1, 0, &mut t1)
        .unwrap();
    let c1: F = t1.challenge();

    // Manual: just coefficients, no label
    let mut t2 = MockTranscript::<F>::default();
    for coeff in poly.coefficients() {
        coeff.append_to_transcript(&mut t2);
    }
    let c2: F = t2.challenge();

    assert_eq!(c1, c2, "unlabeled absorption must match manual absorption");
}

#[test]
fn batched_verify_same_size() {
    // Two polynomials, both 2 variables
    let evals_a: Vec<F> = (1..=4).map(F::from_u64).collect();
    let evals_b: Vec<F> = (5..=8).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

    // Prove: absorb claims, squeeze alpha, combine, prove combined
    let mut pt = MockTranscript::<F>::default();

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

    let mut vt = MockTranscript::<F>::default();
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

    let mut pt = MockTranscript::<F>::default();

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

    let mut vt = MockTranscript::<F>::default();
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
    let mut pt = MockTranscript::<F>::default();

    sum.append_to_transcript(&mut pt);
    let _alpha: F = pt.challenge();

    // alpha^0 = 1, so combined polynomial = evals (single claim)
    let proof = honest_prove(&evals, 3, &mut pt);

    let mut vt = MockTranscript::<F>::default();
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
    let mut vt = MockTranscript::<F>::default();
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

    let mut vt = MockTranscript::<F>::default();
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

/// Native FieldBackend path must produce bit-identical (final_eval, challenges)
/// to the legacy `verify` path on the same proof, with the transcript ending
/// in the same state.
#[test]
fn verify_with_backend_native_matches_legacy() {
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier_backend::Native;

    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);
    let num_vars = 3;

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"native_parity");
    let proof = honest_prove(&evals, num_vars, &mut prover_transcript);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut legacy_transcript = Blake2bTranscript::<F>::new(b"native_parity");
    let clear = ClearRoundVerifier::new();
    let (legacy_eval, legacy_challenges) = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &mut legacy_transcript,
        &clear,
    )
    .unwrap();
    let legacy_post: F = legacy_transcript.challenge();

    let mut backend = Native::<F>::new();
    let mut backend_transcript = backend.new_transcript(b"native_parity");
    let claimed_sum_w = backend.wrap_proof(claim.claimed_sum, "input_claim");
    let (backend_eval_w, backend_challenges_w, backend_challenges_f) =
        SumcheckVerifier::verify_with_backend(
            &mut backend,
            &claim,
            &proof.round_polynomials,
            claimed_sum_w,
            &mut backend_transcript,
            None,
            false,
        )
        .unwrap();
    let backend_post: F = backend_transcript.challenge();

    assert_eq!(backend_eval_w, legacy_eval, "final eval mismatch");
    assert_eq!(
        backend_challenges_f, legacy_challenges,
        "challenges (raw F) mismatch"
    );
    assert_eq!(
        backend_challenges_w, legacy_challenges,
        "challenges (Native scalar = F) mismatch"
    );
    assert_eq!(
        backend_post, legacy_post,
        "post-verify transcript challenges diverged"
    );
}

/// Tracing FieldBackend path must record a graph that, when replayed against
/// the wrapped values seen during verification, reproduces the legacy final
/// evaluation and challenges. The `TracingTranscript` is fed the same Fiat-Shamir
/// bytes as a fresh `Blake2bTranscript`, so the proof is byte-identical and
/// the verifier sees the same challenge values on the wire.
#[test]
fn verify_with_backend_tracing_replays_correctly() {
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier_backend::{replay_trace, Tracing};

    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);
    let num_vars = 3;

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"tracing_replay");
    let proof = honest_prove(&evals, num_vars, &mut prover_transcript);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut legacy_transcript = Blake2bTranscript::<F>::new(b"tracing_replay");
    let clear = ClearRoundVerifier::new();
    let (legacy_eval, _legacy_challenges) = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &mut legacy_transcript,
        &clear,
    )
    .unwrap();

    let mut tracer = Tracing::<F>::new();
    let mut tracer_transcript = tracer.new_transcript(b"tracing_replay");
    let claimed_sum_w = tracer.wrap_proof(claim.claimed_sum, "input_claim");
    let (final_eval_w, _challenges_w, _challenges_f) = SumcheckVerifier::verify_with_backend(
        &mut tracer,
        &claim,
        &proof.round_polynomials,
        claimed_sum_w,
        &mut tracer_transcript,
        None,
        false,
    )
    .unwrap();
    drop(tracer_transcript);

    let graph = tracer.snapshot();
    let wraps = tracer.wrap_values();
    let values = replay_trace(&graph, &wraps).unwrap();
    let final_eval_replayed = values[final_eval_w.id.0 as usize];

    assert_eq!(final_eval_replayed, legacy_eval, "Tracing replay mismatch");
    assert!(
        graph.assertion_count() > 0,
        "Tracing should have recorded sumcheck round-consistency assertions"
    );
    assert!(
        graph
            .nodes
            .iter()
            .any(|n| matches!(n, jolt_verifier_backend::AstOp::TranscriptInit { .. })),
        "Tracing should have recorded a TranscriptInit node"
    );
    assert!(
        graph.nodes.iter().any(|n| matches!(
            n,
            jolt_verifier_backend::AstOp::TranscriptChallengeValue { .. }
        )),
        "Tracing should have recorded TranscriptChallengeValue nodes for sumcheck rounds"
    );
}
