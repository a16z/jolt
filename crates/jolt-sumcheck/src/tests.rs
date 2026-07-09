//! Unit tests for sumcheck verification.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests may panic on assertion failures"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Label, LabelWithCount, Transcript};

use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
use crate::committed::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, CommittedSumcheckConsistency,
    CommittedSumcheckProof, VerifiedCommittedRound,
};
use crate::error::SumcheckError;
use crate::proof::{ClearProof, ClearSumcheckProof, CompressedSumcheckProof, SumcheckProof};
use crate::round_proof::{ClearRound, CompressedLabeledRoundPoly, LabeledRoundPoly, RoundMessage};
use crate::verifier::SumcheckVerifier;
use crate::{
    BatchedCommittedSumcheckConsistency, BooleanHypercube, CenteredIntegerDomain, SumcheckDomain,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};

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
        <UnivariatePoly<F> as RoundMessage>::append_to_transcript(&round_poly, transcript);

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

fn honest_prove_compressed_labeled(
    evals: &[F],
    num_vars: usize,
    label: &'static [u8],
    transcript: &mut Blake2bTranscript,
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
        let compressed = CompressedLabeledRoundPoly::new(&round_poly, label);
        <CompressedLabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(
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

    let mut prover_transcript = Blake2bTranscript::new(b"sumcheck-test");
    let proof = honest_prove(&evals, num_vars, &mut prover_transcript);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut verifier_transcript = Blake2bTranscript::new(b"sumcheck-test");
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

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let proof = honest_prove(&evals, 1, &mut pt);

    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
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

    let mut transcript = Blake2bTranscript::new(b"sumcheck-integer-domain-test");
    let result = SumcheckVerifier::verify(
        &claim,
        std::slice::from_ref(&round_poly),
        CenteredIntegerDomain::new(3),
        &mut transcript,
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

    let mut transcript = Blake2bTranscript::new(b"sumcheck-integer-domain-test");
    let result = SumcheckVerifier::verify(
        &claim,
        &[round_poly],
        CenteredIntegerDomain::new(4),
        &mut transcript,
    );

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

    let mut transcript = Blake2bTranscript::new(b"sumcheck-integer-domain-test");
    let result = SumcheckVerifier::verify(
        &claim,
        &[round_poly],
        CenteredIntegerDomain::new(4),
        &mut transcript,
    );

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

    let mut transcript = Blake2bTranscript::new(b"sumcheck-integer-domain-test");
    let result = SumcheckVerifier::verify(
        &claim,
        &[round_poly],
        CenteredIntegerDomain::new(0),
        &mut transcript,
    );

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

    let mut pt = Blake2bTranscript::new(b"sumcheck-test");
    let proof = honest_prove(&evals, 2, &mut pt);

    // Claim a different sum
    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: real_sum + F::from_u64(1),
    };

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
    let result =
        SumcheckVerifier::verify(&claim, &proof.round_polynomials, BooleanHypercube, &mut vt);
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
    let label: &[u8; 10] = b"test_label";
    let labeled = LabeledRoundPoly::new(&poly, label);

    let mut t1 = Blake2bTranscript::<F>::new(b"sumcheck-test");
    <LabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(&labeled, &mut t1);
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

    let mut t1 = Blake2bTranscript::<F>::new(b"sumcheck-test");
    <UnivariatePoly<F> as RoundMessage>::append_to_transcript(&poly, &mut t1);
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
    let label: &[u8; 15] = b"compressed_test";
    let compressed = CompressedLabeledRoundPoly::new(&poly, label);

    let mut t1 = Blake2bTranscript::<F>::new(b"sumcheck-test");
    <CompressedLabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(&compressed, &mut t1);
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
    let compressed = CompressedLabeledRoundPoly::new(&poly, b"compressed_test");

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
    let compressed = CompressedLabeledRoundPoly::new(&degree_zero, b"compressed_test");

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
    let label = SUMCHECK_ROUND_TRANSCRIPT_LABEL;

    let mut prover_transcript = Blake2bTranscript::new(b"sumcheck-test");
    let (clear_proof, compressed_proof) =
        honest_prove_compressed_labeled(&evals, num_vars, label, &mut prover_transcript);
    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let wrapped = clear_proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, label))
        .collect::<Vec<_>>();
    let mut borrowed_transcript = Blake2bTranscript::new(b"sumcheck-test");
    let borrowed =
        SumcheckVerifier::verify(&claim, &wrapped, BooleanHypercube, &mut borrowed_transcript)
            .unwrap();

    let mut owned_transcript = Blake2bTranscript::new(b"sumcheck-test");
    let owned = compressed_proof
        .verify(&claim, BooleanHypercube, label, &mut owned_transcript)
        .unwrap();

    assert_eq!(owned, borrowed);
    assert_eq!(owned_transcript.state(), borrowed_transcript.state());

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

    let mut transcript = Blake2bTranscript::<F>::new(b"sumcheck-test");
    let result = proof.verify(
        &claim,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );

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

    let mut transcript = Blake2bTranscript::<F>::new(b"sumcheck-test");
    let result = proof.verify(
        &claim,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );

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

    let mut transcript = Blake2bTranscript::<F>::new(b"sumcheck-test");
    let result = proof.verify(
        &claim,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );

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

    let mut transcript = Blake2bTranscript::<F>::new(b"sumcheck-proof-dispatch");
    let reduction = proof
        .verify(
            &claim,
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )
        .unwrap();

    assert_eq!(reduction.point.len(), 1);
}

#[test]
fn sumcheck_proof_verify_rejects_wrong_clear_encoding() {
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Compressed(CompressedSumcheckProof {
        round_polynomials: Vec::new(),
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut transcript = Blake2bTranscript::<F>::new(b"sumcheck-proof-dispatch");
    let result = proof.verify(
        &claim,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );

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

    let mut transcript = Blake2bTranscript::<F>::new(b"sumcheck-proof-dispatch");
    let result = proof.verify(
        &claim,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::WrongProofEncoding {
            expected: "full clear",
            got: "committed",
        })
    ));
}

#[test]
fn batched_committed_consistency_accessors() {
    // `BatchedCommittedSumcheckConsistency` is produced by the generated ZK
    // verify driver (jolt-verifier-derive) and read back by BlindFold through
    // these accessors. Exercise the front-loaded suffix arithmetic and its
    // range errors directly on a hand-built instance.
    let consistency = CommittedSumcheckConsistency {
        rounds: vec![
            VerifiedCommittedRound {
                commitment: F::from_u64(11),
                degree: 2,
                challenge: F::from_u64(101),
            },
            VerifiedCommittedRound {
                commitment: F::from_u64(12),
                degree: 1,
                challenge: F::from_u64(102),
            },
            VerifiedCommittedRound {
                commitment: F::from_u64(13),
                degree: 0,
                challenge: F::from_u64(103),
            },
        ],
    };
    let batched = BatchedCommittedSumcheckConsistency {
        consistency,
        batching_coefficients: vec![F::from_u64(7), F::from_u64(9)],
        max_num_vars: 3,
        max_degree: 2,
    };

    let challenges = vec![F::from_u64(101), F::from_u64(102), F::from_u64(103)];
    assert_eq!(batched.challenges(), challenges);
    assert_eq!(batched.try_round_offset(1).unwrap(), 2);
    assert_eq!(
        batched.try_instance_point(1).unwrap(),
        challenges[2..].to_vec()
    );
    assert_eq!(batched.try_instance_point_at(0, 3).unwrap(), challenges);
    assert!(matches!(
        batched.try_instance_point(4),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 0,
            num_vars: 4,
            total: 3
        })
    ));
    assert!(matches!(
        batched.try_instance_point_at(usize::MAX, 1),
        Err(SumcheckError::BatchedPointRangeOverflow {
            offset: usize::MAX,
            num_vars: 1
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

    let mut manual = Blake2bTranscript::<F>::new(b"committed-sumcheck");
    let mut expected_challenges = Vec::new();
    for round in &rounds {
        manual.append(&Label(b"sumcheck_commitment"));
        round.commitment.append_to_transcript(&mut manual);
        expected_challenges.push(manual.challenge());
    }

    let mut verifier = Blake2bTranscript::<F>::new(b"committed-sumcheck");
    let consistency = SumcheckVerifier::verify_committed_round_consistency(
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
    assert_eq!(verifier.state(), manual.state());
}

#[test]
fn committed_rounds_reject_wrong_round_count() {
    let rounds = vec![CommittedRound {
        commitment: F::from_u64(11),
        degree: 1,
    }];
    let mut transcript = Blake2bTranscript::<F>::new(b"committed-sumcheck");

    let result = SumcheckVerifier::verify_committed_round_consistency(
        SumcheckStatement::new(2, 1),
        &rounds,
        &mut transcript,
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
    let mut transcript = Blake2bTranscript::<F>::new(b"committed-sumcheck");
    let before = transcript.state();

    let result = SumcheckVerifier::verify_committed_round_consistency(
        SumcheckStatement::new(1, 2),
        &rounds,
        &mut transcript,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    assert_eq!(transcript.state(), before);
}

#[test]
fn committed_output_claims_absorb_length_and_order() {
    let output_claims = CommittedOutputClaims {
        commitments: vec![F::from_u64(3), F::from_u64(5), F::from_u64(8)],
    };

    let mut actual = Blake2bTranscript::<F>::new(b"committed-output");
    output_claims.append_to_transcript(&mut actual);

    let mut expected = Blake2bTranscript::<F>::new(b"committed-output");
    expected.append(&LabelWithCount(b"output_claims_coms", 3));
    for commitment in &output_claims.commitments {
        commitment.append_to_transcript(&mut expected);
    }

    assert_eq!(actual.state(), expected.state());
    assert_eq!(actual.challenge(), expected.challenge());
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

    let mut manual = Blake2bTranscript::<F>::new(b"committed-proof");
    let mut expected_challenges = Vec::new();
    for round in &proof.rounds {
        manual.append(&Label(b"sumcheck_commitment"));
        round.commitment.append_to_transcript(&mut manual);
        expected_challenges.push(manual.challenge());
    }
    manual.append(&LabelWithCount(b"output_claims_coms", 2));
    for commitment in &proof.output_claims.commitments {
        commitment.append_to_transcript(&mut manual);
    }

    let mut verifier = Blake2bTranscript::<F>::new(b"committed-proof");
    let consistency = proof
        .verify_committed_consistency(SumcheckStatement::new(2, 2), &mut verifier)
        .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.round_degrees(), vec![1, 2]);
    assert_eq!(verifier.state(), manual.state());
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
    let mut transcript = Blake2bTranscript::<F>::new(b"committed-proof");
    let before = transcript.state();

    let result = proof.verify_committed_consistency(SumcheckStatement::new(1, 2), &mut transcript);

    assert!(matches!(
        result,
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    assert_eq!(transcript.state(), before);
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

impl RoundMessage for MockClearRound {
    fn degree(&self) -> usize {
        0
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        F::from_u64(42).append_to_transcript(transcript);
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

    let mut vt = Blake2bTranscript::new(b"sumcheck-test");
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

/// Drive an honest degree-1 sumcheck through the clear recorder and check the
/// assembled proof against `verify_compressed_boolean` on a twin transcript:
/// the recorder's writes (claim absorb, compressed round polys, opening-claim
/// absorb) must be byte-identical to what the verifier reads back, and the
/// verifier's reduction must land on the prover's bound value.
#[test]
fn clear_recorder_roundtrip_matches_compressed_verifier() {
    use crate::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
    use crate::{append_sumcheck_claim, OPENING_CLAIM_TRANSCRIPT_LABEL};

    let num_vars = 3;
    let evals: Vec<F> = (0..1u64 << num_vars)
        .map(|i| F::from_u64(3 * i + 7))
        .collect();
    let claimed_sum = compute_sum(&evals);

    // Prover side: recorder-driven round loop (HighToLow binding, degree 1).
    let mut prover_transcript = Blake2bTranscript::new(b"recorder-test");
    let mut recorder = ClearSumcheckRecorder::<F, Bn254G1>::new();
    recorder.absorb_input_claims(&[claimed_sum], &mut prover_transcript);

    let mut buf = evals;
    for _round in 0..num_vars {
        let half = buf.len() / 2;
        let eval_0: F = buf[..half].iter().copied().sum();
        let eval_1: F = buf[half..].iter().copied().sum();
        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);

        let r = recorder
            .absorb_round(&round_poly, &mut prover_transcript)
            .unwrap();

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }
    let final_eval = buf[0];
    let recorded = recorder
        .finish(&[final_eval], &mut prover_transcript)
        .unwrap();

    // Verifier side: same transcript schedule via the public verify path.
    let mut verifier_transcript = Blake2bTranscript::new(b"recorder-test");
    append_sumcheck_claim(&mut verifier_transcript, &claimed_sum);
    let reduction = recorded
        .proof
        .verify_compressed_boolean(num_vars, 1, claimed_sum, &mut verifier_transcript)
        .unwrap();
    verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &reduction.value);

    assert_eq!(reduction.value, final_eval);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}
