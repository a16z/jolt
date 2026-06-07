//! Unit tests for sumcheck verification.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests may panic on assertion failures"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::{
    prover_transcript, verifier_transcript, Blake2b512, FsAbsorb, FsChallenge, FsTranscript,
};

use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
use crate::committed::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, CommittedSumcheckProof,
};
use crate::error::SumcheckError;
use crate::proof::{ClearProof, ClearSumcheckProof, CompressedSumcheckProof, SumcheckProof};
use crate::round_proof::{ClearRound, CompressedLabeledRoundPoly, RoundDegree, RoundMessage};
use crate::verifier::SumcheckVerifier;
use crate::{
    append_sumcheck_claim, BatchedSumcheckVerifier, BooleanHypercube, CenteredIntegerDomain,
    SumcheckDomain,
};

type F = Fr;

const INSTANCE: [u8; 32] = [0u8; 32];

/// Build an honest sumcheck proof for a multilinear polynomial given
/// as evaluations over {0,1}^n.
///
/// This is a minimal reference prover: in each round it computes the
/// round polynomial by partial evaluation, absorbs it into the
/// transcript, squeezes a challenge, and binds.
fn honest_prove<T: FsTranscript<F>>(
    evals: &[F],
    num_vars: usize,
    transcript: &mut T,
) -> ClearSumcheckProof<F> {
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

        // Absorb through the same path the unlabelled verifier uses.
        <UnivariatePoly<F> as RoundMessage<F>>::append_to_transcript(&round_poly, transcript);

        let r: F = transcript.challenge();
        round_polys.push(round_poly);

        // Bind HighToLow: buf[i] = buf[i] + r * (buf[i + half] - buf[i])
        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    ClearSumcheckProof {
        round_polynomials: round_polys,
    }
}

fn honest_prove_compressed_labeled<T: FsTranscript<F>>(
    evals: &[F],
    num_vars: usize,
    transcript: &mut T,
) -> (ClearSumcheckProof<F>, CompressedSumcheckProof<F>) {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut compressed_round_polys = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = buf.len() / 2;

        let mut eval_0 = F::from_u64(0);
        let mut eval_1 = F::from_u64(0);
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }

        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);
        let compressed = CompressedLabeledRoundPoly::new(&round_poly);
        <CompressedLabeledRoundPoly<'_, F> as RoundMessage<F>>::append_to_transcript(
            &compressed,
            transcript,
        );

        let r: F = transcript.challenge();
        compressed_round_polys.push(round_poly.compress());
        round_polys.push(round_poly);

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    (
        ClearSumcheckProof {
            round_polynomials: round_polys,
        },
        CompressedSumcheckProof {
            round_polynomials: compressed_round_polys,
        },
    )
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

    let mut prover_transcript =
        prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let proof = honest_prove(&evals, num_vars, &mut prover_transcript);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut verifier_transcript =
        verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        BooleanHypercube,
        &mut verifier_transcript,
    );
    assert!(result.is_ok(), "verification failed: {:?}", result.err());

    let EvaluationClaim {
        point: challenges,
        value: final_eval,
    } = result.unwrap();
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

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let proof = honest_prove(&evals, 1, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let EvaluationClaim {
        point: challenges,
        value: final_eval,
    } = SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt)
        .unwrap();
    assert_eq!(challenges.len(), 1);

    let poly = jolt_poly::Polynomial::new(evals);
    assert_eq!(final_eval, poly.evaluate_and_consume(&challenges));
}

#[test]
fn centered_integer_domain_verifies_round_sum() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 2,
        claimed_sum: F::from_u64(16),
    };

    let mut t = verifier_transcript(
        b"sumcheck-integer-domain-test",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result = SumcheckVerifier::verify(
        &claim,
        std::slice::from_ref(&round_poly),
        CenteredIntegerDomain::new(3),
        &mut t,
    )
    .unwrap();

    assert_eq!(result.point.len(), 1);
    assert_eq!(result.value, round_poly.evaluate(result.point[0]));
}

#[test]
fn centered_integer_domain_uses_core_even_window_convention() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(0), F::from_u64(1)]);
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_i64(2),
    };

    let mut t = verifier_transcript(
        b"sumcheck-integer-domain-test",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result =
        SumcheckVerifier::verify(&claim, &[round_poly], CenteredIntegerDomain::new(4), &mut t);

    assert!(result.is_ok(), "verification failed: {:?}", result.err());
}

#[test]
fn centered_integer_domain_rejects_wrong_sum() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(0), F::from_u64(1)]);
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(3),
    };

    let mut t = verifier_transcript(
        b"sumcheck-integer-domain-test",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result =
        SumcheckVerifier::verify(&claim, &[round_poly], CenteredIntegerDomain::new(4), &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::RoundCheckFailed {
            round: 0,
            expected,
            actual,
        }) if expected == F::from_u64(3) && actual == F::from_i64(2)
    ));
}

#[test]
fn centered_integer_domain_rejects_empty_domain() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(0), F::from_u64(1)]);
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut t = verifier_transcript(
        b"sumcheck-integer-domain-test",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result =
        SumcheckVerifier::verify(&claim, &[round_poly], CenteredIntegerDomain::new(0), &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::InvalidIntegerDomain { domain_size: 0 })
    ));
}

#[test]
fn centered_integer_domain_exposes_power_sums() {
    let domain = CenteredIntegerDomain::new(4);

    assert_eq!(domain.start().unwrap(), -1);
    assert_eq!(domain.power_sums(4).unwrap(), vec![4, 2, 6, 8]);
    assert_eq!(
        <CenteredIntegerDomain as SumcheckDomain<F>>::round_sum_coefficients(&domain, 3).unwrap(),
        vec![
            F::from_u64(4),
            F::from_u64(2),
            F::from_u64(6),
            F::from_u64(8)
        ]
    );
}

#[test]
fn verify_round_check_failure() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Corrupt the first round polynomial
    let bad_coeffs = vec![F::from_u64(999), F::from_u64(1)];
    proof.round_polynomials[0] = UnivariatePoly::new(bad_coeffs);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt);
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

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let mut proof = honest_prove(&evals, 3, &mut pt);

    // Remove the last round
    let _ = proof.round_polynomials.pop();

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt);
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

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let mut proof = honest_prove(&evals, 2, &mut pt);

    // Replace first round poly with a degree-3 polynomial (4 coefficients)
    proof.round_polynomials[0] =
        UnivariatePoly::new(vec![sum, F::from_u64(1), F::from_u64(0), F::from_u64(1)]);

    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1, // degree bound is 1, but we gave degree 3
        claimed_sum: sum,
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt);
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

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let proof = honest_prove(&evals, 2, &mut pt);

    // Claim a different sum
    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: real_sum + F::from_u64(1),
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        SumcheckError::RoundCheckFailed { round: 0, .. }
    ));
}

#[test]
fn clear_round_verifier_no_label() {
    let poly = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);

    let mut t1 = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    <UnivariatePoly<F> as RoundMessage<F>>::append_to_transcript(&poly, &mut t1);
    let c1: F = FsChallenge::<F>::challenge(&mut t1);

    // Manual: the whole coefficient vector as one message, no label.
    let mut t2 = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    t2.absorb_field_slice(poly.coefficients());
    let c2: F = FsChallenge::<F>::challenge(&mut t2);

    assert_eq!(c1, c2, "unlabeled absorption must match manual absorption");
}

#[test]
fn clear_round_verifier_compressed_matches_manual_absorption() {
    // s(X) = 2 + 3*X + 5*X^2  ⇒  s(0) = 2, s(1) = 10, running_sum = 12.
    // Sanity: 2*c0 + c1 + c2 = 2*2 + 3 + 5 = 12.
    let poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let compressed = CompressedLabeledRoundPoly::new(&poly);

    let mut t1 = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    <CompressedLabeledRoundPoly<'_, F> as RoundMessage<F>>::append_to_transcript(
        &compressed,
        &mut t1,
    );
    let ch1: F = FsChallenge::<F>::challenge(&mut t1);

    // Manual absorb matching the compressed wire format: [c0, c2..cd] as one message.
    let mut t2 = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let coeffs = poly.coefficients();
    let mut compressed = vec![coeffs[0]];
    compressed.extend_from_slice(&coeffs[2..]);
    t2.absorb_field_slice(&compressed);
    let ch2: F = FsChallenge::<F>::challenge(&mut t2);

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
    let compressed = CompressedLabeledRoundPoly::new(&poly);

    let result = BooleanHypercube.check_round_sum(0, wrong_running_sum, &compressed);
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
    let compressed = CompressedLabeledRoundPoly::new(&degree_zero);

    let result = BooleanHypercube.check_round_sum(3, F::from_u64(14), &compressed);
    assert!(
        matches!(
            result,
            Err(SumcheckError::CompressedPolynomialTooShort { round: 3, got: 1 })
        ),
        "compressed verifier must reject degree-0 polynomials: {result:?}"
    );
}

#[test]
fn owned_compressed_verify_matches_borrowed_compressed_rounds() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let num_vars = 3;
    let sum = compute_sum(&evals);

    let mut prover_transcript =
        prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    let (clear_proof, compressed_proof) =
        honest_prove_compressed_labeled(&evals, num_vars, &mut prover_transcript);
    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let wrapped = clear_proof
        .round_polynomials
        .iter()
        .map(CompressedLabeledRoundPoly::new)
        .collect::<Vec<_>>();
    let mut borrowed_transcript =
        verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let borrowed =
        SumcheckVerifier::verify(&claim, &wrapped, BooleanHypercube, &mut borrowed_transcript)
            .unwrap();

    let mut owned_transcript =
        verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let owned = compressed_proof
        .verify(&claim, BooleanHypercube, &mut owned_transcript)
        .unwrap();

    assert_eq!(owned, borrowed);
    // Both transcripts consumed identical messages; the next squeezed challenge
    // must agree (the spongefish `state()` accessor the facade exposed is gone).
    let owned_next: F = FsChallenge::<F>::challenge(&mut owned_transcript);
    let borrowed_next: F = FsChallenge::<F>::challenge(&mut borrowed_transcript);
    assert_eq!(owned_next, borrowed_next);

    let poly = jolt_poly::Polynomial::new(evals);
    assert_eq!(owned.value, poly.evaluate_and_consume(&owned.point));
}

#[test]
fn owned_compressed_verify_rejects_wrong_round_count() {
    let proof = CompressedSumcheckProof {
        round_polynomials: Vec::new(),
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut t = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = proof.verify(&claim, BooleanHypercube, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 1,
            got: 0,
        })
    ));
}

#[test]
fn owned_compressed_verify_rejects_degree_bound_exceeded() {
    let proof = CompressedSumcheckProof {
        round_polynomials: vec![CompressedPoly::new(vec![F::from_u64(1), F::from_u64(2)])],
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(3),
    };

    let mut t = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = proof.verify(&claim, BooleanHypercube, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::DegreeBoundExceeded { got: 2, max: 1 })
    ));
}

#[test]
fn owned_compressed_verify_rejects_empty_round_polynomial() {
    let proof = CompressedSumcheckProof {
        round_polynomials: vec![CompressedPoly::new(Vec::new())],
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut t = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = proof.verify(&claim, BooleanHypercube, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::CompressedPolynomialTooShort { round: 0, got: 0 })
    ));
}

#[test]
fn sumcheck_proof_verify_dispatches_full_clear() {
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: vec![UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3)])],
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(7));

    let mut t = verifier_transcript(
        b"sumcheck-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let reduction = proof.verify(&claim, BooleanHypercube, &mut t).unwrap();

    assert_eq!(reduction.point.len(), 1);
}

#[test]
fn sumcheck_proof_verify_rejects_wrong_clear_encoding() {
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Compressed(CompressedSumcheckProof {
        round_polynomials: Vec::new(),
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut t = verifier_transcript(
        b"sumcheck-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result = proof.verify(&claim, BooleanHypercube, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::WrongProofEncoding {
            expected: "full clear",
            got: "compressed clear",
        })
    ));
}

#[test]
fn sumcheck_proof_verify_rejects_committed_encoding() {
    let proof = SumcheckProof::<F, F>::Committed(CommittedSumcheckProof {
        rounds: vec![CommittedRound {
            commitment: F::from_u64(11),
            degree: 1,
        }],
        output_claims: CommittedOutputClaims {
            commitments: vec![F::from_u64(21)],
        },
    });
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut t = verifier_transcript(
        b"sumcheck-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result = proof.verify(&claim, BooleanHypercube, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::WrongProofEncoding {
            expected: "full clear",
            got: "committed",
        })
    ));
}

#[test]
fn batched_verify_same_size() {
    // Two polynomials, both 2 variables
    let evals_a: Vec<F> = (1..=4).map(F::from_u64).collect();
    let evals_b: Vec<F> = (5..=8).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

    // Prove: absorb claims, squeeze alpha, combine, prove combined.
    // The batched verifier absorbs claimed sums via `absorb_field` and squeezes
    // the optimized `alpha`.
    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());

    pt.absorb_field(&sum_a);
    pt.absorb_field(&sum_b);
    let alpha: F = FsChallenge::<F>::challenge(&mut pt);

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

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = BatchedSumcheckVerifier::verify(
        &claims,
        &proof.round_polynomials,
        BooleanHypercube,
        &mut vt,
    );
    assert!(result.is_ok(), "batched verify failed: {:?}", result.err());

    let challenges = result.unwrap().point;
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

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());

    pt.absorb_field(&sum_a);
    pt.absorb_field(&sum_b);
    let alpha: F = FsChallenge::<F>::challenge(&mut pt);

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

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = BatchedSumcheckVerifier::verify(
        &claims,
        &proof.round_polynomials,
        BooleanHypercube,
        &mut vt,
    );
    assert!(result.is_ok(), "batched verify failed: {:?}", result.err());
}

#[test]
fn batched_verify_uses_domain_padding_scale() {
    let sum_a = F::from_u64(0);
    let sum_b = F::from_u64(1);
    let claims = vec![
        SumcheckClaim {
            num_vars: 1,
            degree: 1,
            claimed_sum: sum_a,
        },
        SumcheckClaim {
            num_vars: 0,
            degree: 1,
            claimed_sum: sum_b,
        },
    ];

    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    pt.absorb_field(&sum_a);
    pt.absorb_field(&sum_b);
    let alpha: F = FsChallenge::<F>::challenge(&mut pt);

    let proof = ClearSumcheckProof {
        round_polynomials: vec![UnivariatePoly::new(vec![alpha])],
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = BatchedSumcheckVerifier::verify(
        &claims,
        &proof.round_polynomials,
        CenteredIntegerDomain::new(3),
        &mut vt,
    )
    .unwrap();

    assert_eq!(result.value, alpha);
    assert_eq!(result.point.len(), 1);
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
    let mut pt = prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());

    pt.absorb_field(&sum);
    let _alpha: F = FsChallenge::<F>::challenge(&mut pt);

    // alpha^0 = 1, so combined polynomial = evals (single claim)
    let proof = honest_prove(&evals, 3, &mut pt);

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = BatchedSumcheckVerifier::verify(
        &[claim],
        &proof.round_polynomials,
        BooleanHypercube,
        &mut vt,
    );
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
    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = BatchedSumcheckVerifier::verify(claims, round_proofs, BooleanHypercube, &mut vt);
    assert!(matches!(result, Err(SumcheckError::EmptyClaims)));
}

#[test]
fn batched_compressed_verify_uses_core_batching_statement() {
    let evals_a: Vec<F> = (1..=8).map(F::from_u64).collect();
    let evals_b: Vec<F> = (1..=4).map(F::from_u64).collect();
    let sum_a = compute_sum(&evals_a);
    let sum_b = compute_sum(&evals_b);

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

    let mut prover_transcript =
        prover_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default());
    for claim in &claims {
        append_sumcheck_claim(&mut prover_transcript, &claim.claimed_sum);
    }
    let batching_coefficients = (0..claims.len())
        .map(|_| FsChallenge::<F>::challenge_scalar(&mut prover_transcript))
        .collect::<Vec<_>>();

    let evals_b_extended: Vec<F> = evals_b.iter().flat_map(|&value| [value, value]).collect();
    let combined: Vec<F> = evals_a
        .iter()
        .zip(&evals_b_extended)
        .map(|(&a, &b)| batching_coefficients[0] * a + batching_coefficients[1] * b)
        .collect();
    let (_full_proof, compressed_proof) =
        honest_prove_compressed_labeled(&combined, 3, &mut prover_transcript);

    let mut verifier_transcript =
        verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = BatchedSumcheckVerifier::verify_compressed(
        &claims,
        &compressed_proof,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(result.batching_coefficients, batching_coefficients);
    assert_eq!(result.max_num_vars, 3);
    assert_eq!(result.max_degree, 1);
    assert_eq!(
        result.instance_point(2),
        &result.reduction.point.as_slice()[1..]
    );
    assert_eq!(
        result.try_instance_point(2).unwrap(),
        &result.reduction.point.as_slice()[1..]
    );
    assert_eq!(
        result.try_instance_point_at(0, 3).unwrap(),
        result.reduction.point.as_slice()
    );
    assert!(matches!(
        result.try_instance_point(4),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 0,
            num_vars: 4,
            total: 3
        })
    ));
    assert!(matches!(
        result.try_instance_point_at(usize::MAX, 1),
        Err(SumcheckError::BatchedPointRangeOverflow {
            offset: usize::MAX,
            num_vars: 1
        })
    ));
}

#[test]
fn batched_sumcheck_proof_verify_dispatches_compressed_clear() {
    let evals: Vec<F> = (1..=4).map(F::from_u64).collect();
    let claim = SumcheckClaim::new(2, 1, compute_sum(&evals));

    let mut prover_transcript =
        prover_transcript(b"batched-proof-dispatch", INSTANCE, Blake2b512::default());
    append_sumcheck_claim(&mut prover_transcript, &claim.claimed_sum);
    let _batching_coefficient: F = FsChallenge::<F>::challenge_scalar(&mut prover_transcript);
    let (_full_proof, compressed_proof) =
        honest_prove_compressed_labeled(&evals, 2, &mut prover_transcript);
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Compressed(compressed_proof));

    let mut verifier_transcript = verifier_transcript(
        b"batched-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let reduction = BatchedSumcheckVerifier::verify_compressed_boolean(
        &[claim],
        &proof,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(reduction.max_num_vars, 2);
    assert_eq!(reduction.max_degree, 1);
    assert_eq!(reduction.batching_coefficients.len(), 1);
}

#[test]
fn batched_sumcheck_proof_verify_rejects_full_clear_encoding() {
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: Vec::new(),
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut t = verifier_transcript(
        b"batched-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result = BatchedSumcheckVerifier::verify_compressed_boolean(&[claim], &proof, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::WrongProofEncoding {
            expected: "compressed clear",
            got: "full clear",
        })
    ));
}

#[test]
fn batched_committed_consistency_uses_statements_without_clear_claims() {
    let statements = [SumcheckStatement::new(3, 2), SumcheckStatement::new(1, 1)];
    let proof = SumcheckProof::<F, F>::Committed(CommittedSumcheckProof {
        rounds: vec![
            CommittedRound {
                commitment: F::from_u64(11),
                degree: 2,
            },
            CommittedRound {
                commitment: F::from_u64(12),
                degree: 1,
            },
            CommittedRound {
                commitment: F::from_u64(13),
                degree: 0,
            },
        ],
        output_claims: CommittedOutputClaims {
            commitments: vec![F::from_u64(21), F::from_u64(34)],
        },
    });

    let mut manual = prover_transcript(b"batched-proof-dispatch", INSTANCE, Blake2b512::default());
    let batching_coefficients = (0..statements.len())
        .map(|_| FsChallenge::<F>::challenge_scalar(&mut manual))
        .collect::<Vec<_>>();
    let mut expected_challenges = Vec::new();
    let SumcheckProof::Committed(committed_proof) = &proof else {
        panic!("proof must be committed");
    };
    for round in &committed_proof.rounds {
        manual.absorb(&round.commitment);
        expected_challenges.push(FsChallenge::<F>::challenge(&mut manual));
    }
    manual.absorb(&committed_proof.output_claims.commitments);

    let mut verifier = verifier_transcript(
        b"batched-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let consistency =
        BatchedSumcheckVerifier::verify_committed_consistency(&statements, &proof, &mut verifier)
            .unwrap();

    assert_eq!(consistency.batching_coefficients, batching_coefficients);
    assert_eq!(consistency.max_num_vars, 3);
    assert_eq!(consistency.max_degree, 2);
    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.try_round_offset(1).unwrap(), 2);
    assert_eq!(
        consistency.try_instance_point(1).unwrap(),
        expected_challenges[2..].to_vec()
    );
    assert_eq!(
        consistency.try_instance_point_at(0, 3).unwrap(),
        expected_challenges
    );
    assert!(matches!(
        consistency.try_instance_point(4),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 0,
            num_vars: 4,
            total: 3
        })
    ));
    assert!(matches!(
        consistency.try_instance_point_at(usize::MAX, 1),
        Err(SumcheckError::BatchedPointRangeOverflow {
            offset: usize::MAX,
            num_vars: 1
        })
    ));
    // Both transcripts consumed identical messages; the next squeezed challenge
    // must agree.
    let verifier_next: F = FsChallenge::<F>::challenge(&mut verifier);
    let manual_next: F = FsChallenge::<F>::challenge(&mut manual);
    assert_eq!(verifier_next, manual_next);
}

#[test]
fn batched_claim_verifier_rejects_committed_encoding() {
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));
    let proof = SumcheckProof::<F, F>::Committed(CommittedSumcheckProof {
        rounds: vec![CommittedRound {
            commitment: F::from_u64(11),
            degree: 1,
        }],
        output_claims: CommittedOutputClaims::default(),
    });

    let mut t = verifier_transcript(
        b"batched-proof-dispatch",
        INSTANCE,
        Blake2b512::default(),
        &[],
    );
    let result = BatchedSumcheckVerifier::verify_compressed_boolean(&[claim], &proof, &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::WrongProofEncoding {
            expected: "compressed clear",
            got: "committed",
        })
    ));
}

#[test]
fn committed_rounds_check_transcript_and_return_public_data() {
    let rounds = vec![
        CommittedRound {
            commitment: F::from_u64(11),
            degree: 1,
        },
        CommittedRound {
            commitment: F::from_u64(12),
            degree: 2,
        },
        CommittedRound {
            commitment: F::from_u64(13),
            degree: 0,
        },
    ];

    let mut manual = prover_transcript(b"committed-sumcheck", INSTANCE, Blake2b512::default());
    let mut expected_challenges = Vec::new();
    for round in &rounds {
        manual.absorb(&round.commitment);
        expected_challenges.push(FsChallenge::<F>::challenge(&mut manual));
    }

    let mut verifier =
        verifier_transcript(b"committed-sumcheck", INSTANCE, Blake2b512::default(), &[]);
    let consistency = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(3, 2),
        &rounds,
        &mut verifier,
    )
    .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.round_degrees(), vec![1, 2, 0]);
    assert_eq!(
        consistency.round_commitments(),
        rounds
            .iter()
            .map(|round| round.commitment)
            .collect::<Vec<_>>()
    );
    let verifier_next: F = FsChallenge::<F>::challenge(&mut verifier);
    let manual_next: F = FsChallenge::<F>::challenge(&mut manual);
    assert_eq!(verifier_next, manual_next);
}

#[test]
fn committed_rounds_reject_wrong_round_count() {
    let rounds = vec![CommittedRound {
        commitment: F::from_u64(11),
        degree: 1,
    }];
    let mut t = verifier_transcript(b"committed-sumcheck", INSTANCE, Blake2b512::default(), &[]);

    let result = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 1),
        &rounds,
        &mut t,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn committed_rounds_reject_degree_bound_before_absorbing() {
    let rounds = vec![CommittedRound {
        commitment: F::from_u64(11),
        degree: 3,
    }];
    let mut t = verifier_transcript(b"committed-sumcheck", INSTANCE, Blake2b512::default(), &[]);
    // Capture the challenge that would be squeezed if nothing had been absorbed.
    let mut probe =
        verifier_transcript(b"committed-sumcheck", INSTANCE, Blake2b512::default(), &[]);
    let before: F = FsChallenge::<F>::challenge(&mut probe);

    let result = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(1, 2),
        &rounds,
        &mut t,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    // The degree check rejects before absorbing, so the transcript is untouched
    // and still squeezes the pristine challenge.
    let after: F = FsChallenge::<F>::challenge(&mut t);
    assert_eq!(after, before);
}

#[test]
fn committed_output_claims_absorb_length_and_order() {
    let output_claims = CommittedOutputClaims {
        commitments: vec![F::from_u64(3), F::from_u64(5), F::from_u64(8)],
    };

    let mut actual = prover_transcript(b"committed-output", INSTANCE, Blake2b512::default());
    output_claims.append_to_transcript(&mut actual);

    let mut expected = prover_transcript(b"committed-output", INSTANCE, Blake2b512::default());
    expected.absorb(&output_claims.commitments);

    let actual_next: F = FsChallenge::<F>::challenge(&mut actual);
    let expected_next: F = FsChallenge::<F>::challenge(&mut expected);
    assert_eq!(actual_next, expected_next);
}

#[test]
fn committed_proof_checks_rounds_then_output_claims() {
    let proof = CommittedSumcheckProof {
        rounds: vec![
            CommittedRound {
                commitment: F::from_u64(11),
                degree: 1,
            },
            CommittedRound {
                commitment: F::from_u64(12),
                degree: 2,
            },
        ],
        output_claims: CommittedOutputClaims {
            commitments: vec![F::from_u64(21), F::from_u64(34)],
        },
    };

    let mut manual = prover_transcript(b"committed-proof", INSTANCE, Blake2b512::default());
    let mut expected_challenges = Vec::new();
    for round in &proof.rounds {
        manual.absorb(&round.commitment);
        expected_challenges.push(FsChallenge::<F>::challenge(&mut manual));
    }
    manual.absorb(&proof.output_claims.commitments);

    let mut verifier =
        verifier_transcript(b"committed-proof", INSTANCE, Blake2b512::default(), &[]);
    let consistency = proof
        .verify_committed_consistency::<F, _>(SumcheckStatement::new(2, 2), &mut verifier)
        .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.round_degrees(), vec![1, 2]);
    let verifier_next: F = FsChallenge::<F>::challenge(&mut verifier);
    let manual_next: F = FsChallenge::<F>::challenge(&mut manual);
    assert_eq!(verifier_next, manual_next);
}

#[test]
fn committed_proof_rejects_bad_round_before_output_claims() {
    let proof = CommittedSumcheckProof {
        rounds: vec![CommittedRound {
            commitment: F::from_u64(11),
            degree: 3,
        }],
        output_claims: CommittedOutputClaims {
            commitments: vec![F::from_u64(21)],
        },
    };
    let mut t = verifier_transcript(b"committed-proof", INSTANCE, Blake2b512::default(), &[]);
    let mut probe = verifier_transcript(b"committed-proof", INSTANCE, Blake2b512::default(), &[]);
    let before: F = FsChallenge::<F>::challenge(&mut probe);

    let result = proof.verify_committed_consistency::<F, _>(SumcheckStatement::new(1, 2), &mut t);

    assert!(matches!(
        result,
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    let after: F = FsChallenge::<F>::challenge(&mut t);
    assert_eq!(after, before);
}

#[test]
fn committed_round_witness_commits_with_generic_vector_commitment() {
    type VC = Pedersen<Bn254G1>;

    let generator = Bn254::g1_generator();
    let setup = PedersenSetup::new(
        vec![
            generator,
            generator.scalar_mul(&F::from_u64(2)),
            generator.scalar_mul(&F::from_u64(3)),
        ],
        generator.scalar_mul(&F::from_u64(99)),
    );
    let witness = CommittedRoundWitness {
        coefficients: vec![F::from_u64(4), F::from_u64(5), F::from_u64(6)],
        blinding: F::from_u64(7),
    };

    let round = witness.commit::<VC>(&setup).unwrap();

    assert_eq!(round.degree, 2);
    assert!(VC::verify(
        &setup,
        &round.commitment,
        &witness.coefficients,
        &witness.blinding
    ));
    assert!(!VC::verify(
        &setup,
        &round.commitment,
        &witness.coefficients,
        &(witness.blinding + F::from_u64(1))
    ));
}

#[test]
fn committed_round_witness_rejects_empty_coefficients() {
    type VC = Pedersen<Bn254G1>;

    let generator = Bn254::g1_generator();
    let setup = PedersenSetup::new(vec![generator], generator.scalar_mul(&F::from_u64(99)));
    let witness = CommittedRoundWitness {
        coefficients: Vec::new(),
        blinding: F::from_u64(7),
    };

    let result = witness.commit::<VC>(&setup);

    assert!(matches!(result, Err(SumcheckError::EmptyRoundCoefficients)));
}

/// A mock clear round that returns a fixed evaluation.
/// Used to verify that `SumcheckVerifier` accepts custom clear round messages.
struct MockClearRound {
    fixed_sum: F,
}

impl RoundDegree for MockClearRound {
    fn degree(&self) -> usize {
        0
    }
}

impl RoundMessage<F> for MockClearRound {
    fn append_to_transcript<T: FsTranscript<F>>(&self, transcript: &mut T) {
        transcript.absorb_field(&F::from_u64(42));
    }
}

impl ClearRound<F> for MockClearRound {
    fn evaluate(&self, _challenge: F) -> F {
        self.fixed_sum
    }

    fn coefficient_linear_combination(&self, coefficients: &[F]) -> F {
        self.fixed_sum * coefficients[0]
    }
}

#[test]
fn verify_accepts_custom_clear_round_messages() {
    let fixed = F::from_u64(0);
    let round_proofs = [
        MockClearRound { fixed_sum: fixed },
        MockClearRound { fixed_sum: fixed },
        MockClearRound { fixed_sum: fixed },
    ];

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut vt = verifier_transcript(b"sumcheck-test", INSTANCE, Blake2b512::default(), &[]);
    let result = SumcheckVerifier::verify(&claim, &round_proofs, BooleanHypercube, &mut vt);
    assert!(
        result.is_ok(),
        "mock verifier should accept: {:?}",
        result.err()
    );

    let EvaluationClaim {
        point: challenges,
        value: final_eval,
    } = result.unwrap();
    assert_eq!(challenges.len(), 3);
    // Mock always returns fixed_sum, so final eval should be fixed
    assert_eq!(final_eval, fixed);
}

#[test]
#[should_panic(expected = "degree >= 1")]
fn sumcheck_claim_new_rejects_degree_zero() {
    let _ = SumcheckClaim::<Fr>::new(3, 0, Fr::from_u64(0));
}

#[test]
#[should_panic(expected = "degree >= 1")]
fn sumcheck_statement_new_rejects_degree_zero() {
    let _ = SumcheckStatement::new(3, 0);
}
