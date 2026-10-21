#![expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::wildcard_enum_match_arm,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures, index directly, and match errors with catch-all arms"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Field, Fr, Prime128OffsetA7F7, Ring};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::{
    prover_transcript, Blake2b512, Encoding, FieldTranscript, LabelWithCount, ProverState,
    ProverTranscript, Transcript,
};

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
    round_pow_bits, BatchedCommittedSumcheckConsistency, BooleanHypercube, CenteredIntegerDomain,
    Nonced, NoncedRef, SumcheckDomain, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};

type F = Fr;
/// BN254 challenges need no proof of work, so the proof-of-work tests run
/// over Akita's field, whose rounds grind.
type P = Prime128OffsetA7F7;
type PT = ProverState<Blake2b512>;

fn pow_bits(degree: usize) -> u32 {
    round_pow_bits::<F, ProverState<Blake2b512>>(degree).unwrap()
}

/// Replays the verifier's round loop on `transcript` — absorb, proof of work,
/// challenge — and returns the nonces it grinds at the difficulty for `degree`.
fn grind_rounds<R: RoundMessage>(
    rounds: &[R],
    degree: usize,
    transcript: &mut ProverState<Blake2b512>,
) -> Vec<u32> {
    rounds
        .iter()
        .filter_map(|round| {
            round.append_to_transcript(transcript);
            let nonce = transcript.public_grind(pow_bits(degree)).unwrap();
            let _challenge: F = transcript.challenge().unwrap();
            nonce
        })
        .collect()
}

fn honest_prove<E: Field + Encoding, T: FieldTranscript<E> + ProverTranscript>(
    evals: &[E],
    num_vars: usize,
    transcript: &mut T,
) -> ClearSumcheckProof<E> {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut pow_nonces = Vec::new();

    for _round in 0..num_vars {
        let half = buf.len() / 2;

        let mut eval_0 = E::from_u64(0);
        let mut eval_1 = E::from_u64(0);
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }

        let c0 = eval_0;
        let c1 = eval_1 - eval_0;
        let round_poly = UnivariatePoly::new(vec![c0, c1]);

        // Absorb through the same path the unlabelled verifier uses.
        <UnivariatePoly<E> as RoundMessage>::append_to_transcript(&round_poly, transcript);
        pow_nonces.extend(
            transcript
                .public_grind(round_pow_bits::<E, T>(1).unwrap())
                .unwrap(),
        );

        let r: E = transcript.challenge().unwrap();
        round_polys.push(round_poly);

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    ClearSumcheckProof {
        round_polynomials: round_polys,
        pow_nonces,
    }
}

fn honest_prove_compressed_labeled<
    E: Field + Encoding,
    T: FieldTranscript<E> + ProverTranscript,
>(
    evals: &[E],
    num_vars: usize,
    label: &'static [u8],
    transcript: &mut T,
) -> (ClearSumcheckProof<E>, CompressedSumcheckProof<E>) {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut compressed_round_polys = Vec::with_capacity(num_vars);
    let mut pow_nonces = Vec::new();

    for _round in 0..num_vars {
        let half = buf.len() / 2;

        let mut eval_0 = E::from_u64(0);
        let mut eval_1 = E::from_u64(0);
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }

        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);
        let compressed = CompressedLabeledRoundPoly::new(&round_poly, label);
        <CompressedLabeledRoundPoly<'_, E> as RoundMessage>::append_to_transcript(
            &compressed,
            transcript,
        );
        pow_nonces.extend(
            transcript
                .public_grind(round_pow_bits::<E, T>(1).unwrap())
                .unwrap(),
        );

        let r: E = transcript.challenge().unwrap();
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
            pow_nonces: pow_nonces.clone(),
        },
        CompressedSumcheckProof {
            round_polynomials: compressed_round_polys,
            pow_nonces,
        },
    )
}

fn compute_sum<E: Ring>(evals: &[E]) -> E {
    evals.iter().copied().sum()
}

#[test]
fn verify_valid_degree1_proof() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let sum = compute_sum(&evals);
    let num_vars = 3;

    let mut prover_state = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let proof = honest_prove(&evals, num_vars, &mut prover_state);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &proof.pow_nonces,
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
fn centered_integer_domain_verifies_round_sum() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 2,
        claimed_sum: F::from_u64(16),
    };

    let pow_nonces = grind_rounds(
        std::slice::from_ref(&round_poly),
        2,
        &mut prover_transcript::<Blake2b512>(b"sumcheck-integer-domain-test", b""),
    );
    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-integer-domain-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        std::slice::from_ref(&round_poly),
        &pow_nonces,
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

    let pow_nonces = grind_rounds(
        std::slice::from_ref(&round_poly),
        1,
        &mut prover_transcript::<Blake2b512>(b"sumcheck-integer-domain-test", b""),
    );
    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-integer-domain-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &[round_poly],
        &pow_nonces,
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

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-integer-domain-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &[round_poly],
        &[],
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

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-integer-domain-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &[round_poly],
        &[],
        CenteredIntegerDomain::new(0),
        &mut transcript,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::InvalidIntegerDomain { domain_size: 0 })
    ));
}

#[test]
fn oversized_integer_domain_is_rejected_without_a_full_scan() {
    let domain_size = i64::MAX as usize;
    let result: Result<Vec<Fr>, _> =
        CenteredIntegerDomain::new(domain_size).round_sum_coefficients(3);
    assert!(matches!(
        result,
        Err(SumcheckError::InvalidIntegerDomain { domain_size: size }) if size == domain_size
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

    let mut pt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let mut proof = honest_prove(&evals, 3, &mut pt);

    let bad_coeffs = vec![F::from_u64(999), F::from_u64(1)];
    proof.round_polynomials[0] = UnivariatePoly::new(bad_coeffs);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &proof.pow_nonces,
        BooleanHypercube,
        &mut vt,
    );
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

    let mut pt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let mut proof = honest_prove(&evals, 3, &mut pt);

    let _ = proof.round_polynomials.pop();

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &proof.pow_nonces,
        BooleanHypercube,
        &mut vt,
    );
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

    let mut pt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let mut proof = honest_prove(&evals, 2, &mut pt);

    proof.round_polynomials[0] =
        UnivariatePoly::new(vec![sum, F::from_u64(1), F::from_u64(0), F::from_u64(1)]);

    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &proof.pow_nonces,
        BooleanHypercube,
        &mut vt,
    );
    match result.unwrap_err() {
        SumcheckError::DegreeBoundExceeded { got, max } => {
            assert_eq!(max, 1);
            assert!(got > 1);
        }
        other => panic!("expected DegreeBoundExceeded, got {other:?}"),
    }
}

#[test]
fn clear_round_verifier_with_label_absorbs_label() {
    let poly = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);
    let label: &[u8; 10] = b"test_label";
    let labeled = LabeledRoundPoly::new(&poly, label);

    let mut t1 = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    <LabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(&labeled, &mut t1);
    let c1: F = t1.challenge().unwrap();

    let mut t2 = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    t2.append(&LabelWithCount(label, 2));
    for coeff in poly.coefficients() {
        t2.append(&coeff);
    }
    let c2: F = t2.challenge().unwrap();

    assert_eq!(c1, c2, "labeled absorption must match manual absorption");
}

#[test]
fn clear_round_verifier_no_label() {
    let poly = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);

    let mut t1 = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    <UnivariatePoly<F> as RoundMessage>::append_to_transcript(&poly, &mut t1);
    let c1: F = t1.challenge().unwrap();

    let mut t2 = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    for coeff in poly.coefficients() {
        t2.append(&coeff);
    }
    let c2: F = t2.challenge().unwrap();

    assert_eq!(c1, c2, "unlabeled absorption must match manual absorption");
}

#[test]
fn clear_round_verifier_compressed_matches_manual_absorption() {
    let poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let label: &[u8; 15] = b"compressed_test";
    let compressed = CompressedLabeledRoundPoly::new(&poly, label);

    let mut t1 = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    <CompressedLabeledRoundPoly<'_, F> as RoundMessage>::append_to_transcript(&compressed, &mut t1);
    let ch1: F = t1.challenge().unwrap();

    // Manual absorb matching the compressed wire format: label_with_count(d), c0, c2..cd.
    let mut t2 = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let coeffs = poly.coefficients();
    t2.append(&LabelWithCount(label, (coeffs.len() - 1) as u64));
    t2.append(&coeffs[0]);
    for c in coeffs.iter().skip(2) {
        t2.append(&c);
    }
    let ch2: F = t2.challenge().unwrap();

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

    let mut prover_state = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let (clear_proof, compressed_proof) =
        honest_prove_compressed_labeled(&evals, num_vars, label, &mut prover_state);
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
    let mut borrowed_transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let borrowed = SumcheckVerifier::verify(
        &claim,
        &wrapped,
        &compressed_proof.pow_nonces,
        BooleanHypercube,
        &mut borrowed_transcript,
    )
    .unwrap();

    let mut owned_transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let owned = compressed_proof
        .verify(&claim, BooleanHypercube, label, &mut owned_transcript)
        .unwrap();

    assert_eq!(owned, borrowed);
    assert_eq!(
        owned_transcript.verifier_message::<[u8; 32]>(),
        borrowed_transcript.verifier_message::<[u8; 32]>()
    );

    let poly = jolt_poly::Polynomial::new(evals);
    assert_eq!(owned.value, poly.evaluate_and_consume(&owned.point));
}

#[test]
fn owned_compressed_verify_rejects_wrong_round_count() {
    let proof = CompressedSumcheckProof {
        round_polynomials: Vec::new(),
        pow_nonces: Vec::new(),
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
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
        pow_nonces: Vec::new(),
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(3),
    };

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
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
        pow_nonces: Vec::new(),
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
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
    let round_poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3)]);
    let pow_nonces = grind_rounds(
        &[LabeledRoundPoly::sumcheck(&round_poly)],
        1,
        &mut prover_transcript::<Blake2b512>(b"sumcheck-proof-dispatch", b""),
    );
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: vec![round_poly],
        pow_nonces,
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(7));

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-proof-dispatch", b"");
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
        pow_nonces: Vec::new(),
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-proof-dispatch", b"");
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
        pow_nonces: Vec::new(),
        output_claims: CommittedOutputClaims {
            commitments: vec![F::from_u64(21)],
        },
    });
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-proof-dispatch", b"");
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

fn batched_consistency_with_challenges(
    challenges: &[F],
    max_num_vars: usize,
) -> BatchedCommittedSumcheckConsistency<F, F> {
    BatchedCommittedSumcheckConsistency {
        consistency: CommittedSumcheckConsistency {
            rounds: challenges
                .iter()
                .enumerate()
                .map(|(index, &challenge)| VerifiedCommittedRound {
                    commitment: F::from_u64(50 + index as u64),
                    degree: 1,
                    challenge,
                })
                .collect(),
        },
        batching_coefficients: vec![F::from_u64(7)],
        max_num_vars,
        max_degree: 1,
    }
}

#[test]
fn batched_try_round_offset_rejects_instance_wider_than_batch() {
    let challenges: Vec<F> = (101..=103).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 3);

    assert!(matches!(
        batched.try_round_offset(4),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 0,
            num_vars: 4,
            total: 3
        })
    ));
    assert_eq!(batched.try_round_offset(3).unwrap(), 0);
}

#[test]
fn batched_instance_points_are_challenge_suffixes_for_mixed_arities() {
    let challenges: Vec<F> = (201..=204).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 4);

    // Tail-aligned members of a mixed-arity batch (4, 2, and 1 rounds) get
    // dummy rounds front-loaded, so each point is a challenge suffix.
    assert_eq!(batched.try_round_offset(4).unwrap(), 0);
    assert_eq!(batched.try_round_offset(2).unwrap(), 2);
    assert_eq!(batched.try_round_offset(1).unwrap(), 3);
    assert_eq!(batched.try_instance_point(4).unwrap(), challenges);
    assert_eq!(
        batched.try_instance_point(2).unwrap(),
        challenges[2..].to_vec()
    );
    assert_eq!(
        batched.try_instance_point(1).unwrap(),
        challenges[3..].to_vec()
    );
    assert_eq!(
        batched.try_instance_point_at(0, 2).unwrap(),
        challenges[..2].to_vec()
    );
    assert_eq!(
        batched.try_instance_point_at(1, 2).unwrap(),
        challenges[1..3].to_vec()
    );
}

#[test]
fn batched_instance_point_at_rejects_windows_past_recorded_challenges() {
    let challenges: Vec<F> = (301..=303).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 3);

    assert!(matches!(
        batched.try_instance_point_at(2, 2),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 2,
            num_vars: 2,
            total: 3
        })
    ));
    assert!(matches!(
        batched.try_instance_point_at(4, 1),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 4,
            num_vars: 1,
            total: 3
        })
    ));
    assert!(matches!(
        batched.try_instance_point_at(usize::MAX - 1, 2),
        Err(SumcheckError::BatchedPointRangeOverflow {
            offset,
            num_vars: 2
        }) if offset == usize::MAX - 1
    ));
    assert_eq!(
        batched.try_instance_point_at(3, 0).unwrap(),
        Vec::<F>::new()
    );
}

#[test]
fn batched_instance_point_rejects_declared_width_exceeding_recorded_rounds() {
    // Adversarial mismatch: the batch declares five rounds but the proof only
    // recorded three. The suffix offset is computed from the declared width,
    // so the lookup must fail against the recorded challenge count.
    let challenges: Vec<F> = (401..=403).map(F::from_u64).collect();
    let batched = batched_consistency_with_challenges(&challenges, 5);

    assert_eq!(batched.try_round_offset(1).unwrap(), 4);
    assert!(matches!(
        batched.try_instance_point(1),
        Err(SumcheckError::BatchedPointOutOfRange {
            offset: 4,
            num_vars: 1,
            total: 3
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

    let mut manual = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    let mut expected_challenges: Vec<F> = Vec::new();
    let mut pow_nonces = Vec::new();
    for round in &rounds {
        manual.append(&LabelWithCount(b"sumcheck_commitment", round.degree as u64));
        manual.append(&round.commitment);
        // Priced at the statement's degree bound (2), not the round's own degree.
        pow_nonces.extend(manual.public_grind(pow_bits(2)).unwrap());
        expected_challenges.push(manual.challenge().unwrap());
    }

    let mut verifier = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    let consistency = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(3, 2),
        &rounds,
        &pow_nonces,
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
    assert_eq!(
        verifier.verifier_message::<[u8; 32]>(),
        manual.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn committed_rounds_reject_wrong_round_count() {
    let rounds = vec![CommittedRound {
        commitment: F::from_u64(11),
        degree: 1,
    }];
    let mut transcript = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");

    let result = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 1),
        &rounds,
        &[],
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
    let mut transcript = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    let mut untouched = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");

    let result = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(1, 2),
        &rounds,
        &[],
        &mut transcript,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    assert_eq!(
        transcript.verifier_message::<[u8; 32]>(),
        untouched.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn committed_output_claims_absorb_length_and_order() {
    let output_claims = CommittedOutputClaims {
        commitments: vec![F::from_u64(3), F::from_u64(5), F::from_u64(8)],
    };

    let mut actual = prover_transcript::<Blake2b512>(b"committed-output", b"");
    actual.append(&output_claims);

    let mut expected = prover_transcript::<Blake2b512>(b"committed-output", b"");
    expected.append(&LabelWithCount(b"output_claims_coms", 3));
    for commitment in &output_claims.commitments {
        expected.append(&commitment);
    }

    assert_eq!(
        actual.verifier_message::<[u8; 32]>(),
        expected.verifier_message::<[u8; 32]>()
    );
    let actual: F = actual.challenge().unwrap();
    let expected: F = expected.challenge().unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn committed_proof_checks_rounds_then_output_claims() {
    let mut proof = CommittedSumcheckProof {
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
        pow_nonces: Vec::new(),
        output_claims: CommittedOutputClaims {
            commitments: vec![F::from_u64(21), F::from_u64(34)],
        },
    };

    let mut manual = prover_transcript::<Blake2b512>(b"committed-proof", b"");
    let mut expected_challenges: Vec<F> = Vec::new();
    for round in &proof.rounds {
        manual.append(&LabelWithCount(b"sumcheck_commitment", round.degree as u64));
        manual.append(&round.commitment);
        proof
            .pow_nonces
            .extend(manual.public_grind(pow_bits(2)).unwrap());
        expected_challenges.push(manual.challenge().unwrap());
    }
    manual.append(&LabelWithCount(b"output_claims_coms", 2));
    for commitment in &proof.output_claims.commitments {
        manual.append(&commitment);
    }

    let mut verifier = prover_transcript::<Blake2b512>(b"committed-proof", b"");
    let consistency: CommittedSumcheckConsistency<F, F> = proof
        .verify_committed_consistency(SumcheckStatement::new(2, 2), &mut verifier)
        .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.round_degrees(), vec![1, 2]);
    assert_eq!(
        verifier.verifier_message::<[u8; 32]>(),
        manual.verifier_message::<[u8; 32]>()
    );
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

struct MockClearRound {
    fixed_sum: F,
}

impl RoundMessage for MockClearRound {
    fn degree(&self) -> usize {
        0
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&F::from_u64(42));
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

    let pow_nonces = grind_rounds(
        &round_proofs,
        1,
        &mut prover_transcript::<Blake2b512>(b"sumcheck-test", b""),
    );
    let mut vt = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &round_proofs,
        &pow_nonces,
        BooleanHypercube,
        &mut vt,
    );
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

    let mut prover_state = prover_transcript::<Blake2b512>(b"recorder-test", b"");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[claimed_sum], &mut prover_state);

    let mut buf = evals;
    for _round in 0..num_vars {
        let half = buf.len() / 2;
        let eval_0: F = buf[..half].iter().copied().sum();
        let eval_1: F = buf[half..].iter().copied().sum();
        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);

        let r = recorder
            .absorb_round(&round_poly, 1, &mut prover_state)
            .unwrap();

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }
    let final_eval = buf[0];
    let recorded = recorder.finish(&[final_eval], &mut prover_state).unwrap();

    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"recorder-test", b"");
    append_sumcheck_claim(&mut verifier_transcript, &claimed_sum);
    let reduction = SumcheckProof::<F, Bn254G1>::from(recorded.proof)
        .verify_compressed_boolean(num_vars, 1, claimed_sum, &mut verifier_transcript)
        .unwrap();
    verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &reduction.value);

    assert_eq!(reduction.value, final_eval);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// A minimal dense multilinear [`ProveRounds`](crate::prover::ProveRounds)
/// member (HighToLow binding), constructible with a prescribed total sum —
/// the engine tests' stand-in for a real kernel-backed batch member.
pub(crate) struct DenseMember {
    evals: Vec<F>,
    num_rounds: usize,
}

impl DenseMember {
    pub(crate) fn with_sum(num_rounds: usize, sum: F, seed: u64) -> Self {
        let size = 1u64 << num_rounds;
        let mut evals: Vec<F> = (0..size).map(|i| F::from_u64(seed + 31 * i + 11)).collect();
        let current: F = evals.iter().copied().sum();
        evals[0] += sum - current;
        Self { evals, num_rounds }
    }

    fn final_eval(&self) -> F {
        self.evals[0]
    }
}

impl DenseMember {
    fn bind(&mut self, challenge: F) {
        let half = self.evals.len() / 2;
        for i in 0..half {
            self.evals[i] = self.evals[i] + challenge * (self.evals[i + half] - self.evals[i]);
        }
        self.evals.truncate(half);
    }
}

impl crate::prover::ProveRounds<F> for DenseMember {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.evals.len() / 2;
        let eval_0: F = self.evals[..half].iter().copied().sum();
        let eval_1: F = self.evals[half..].iter().copied().sum();
        assert_eq!(eval_0 + eval_1, previous_claim);
        Ok(UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

fn pedersen_setup(capacity: u64) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let generators = (2..2 + capacity)
        .map(|k| generator.scalar_mul(&F::from_u64(k)))
        .collect();
    PedersenSetup::new(generators, generator.scalar_mul(&F::from_u64(99)))
}

/// A batch that declares `max_degree == 0` while having rounds to prove must
/// be rejected with `ZeroBatchDegree`, not proved (a degree-0 round polynomial
/// cannot carry a sumcheck round) and not panic.
#[test]
fn prove_batch_rejects_zero_max_degree() {
    use crate::batch::{BatchMember, BatchPrelude};
    use crate::prover::{prove_batch, ProveRounds, SequentialRounds};
    use crate::recorder::ClearSumcheckRecorder;

    let sum = F::from_u64(42);
    let mut member = DenseMember::with_sum(2, sum, 7);
    let prelude = BatchPrelude {
        members: vec![BatchMember {
            input_claim: sum,
            coefficient: F::from_u64(1),
            rounds: 2,
            offset: 0,
        }],
        claimed_sum: sum,
        max_num_vars: 2,
        max_degree: 0,
    };
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut member];
    let mut transcript = prover_transcript::<Blake2b512>(b"zero-degree-batch", b"");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    let result = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(SumcheckError::ZeroBatchDegree { max_num_vars: 2 })
    ));
}

/// Twin-transcript lock for the batched engine, padding included: a
/// 3-round and a 1-round member proved through `prove_batch` with the clear
/// recorder must be byte-identical to the verifier's head-replica +
/// `verify_compressed_boolean` + opening-claim absorbs, and the reduction must
/// land on the prover's final claim and challenges.
#[test]
fn prove_batch_clear_twin_matches_compressed_verifier_with_padding() {
    use crate::batch::{BatchMember, BatchPrelude};
    use crate::prover::{prove_batch, ProveRounds, SequentialRounds};
    use crate::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
    use crate::{append_sumcheck_claim, OPENING_CLAIM_TRANSCRIPT_LABEL};
    use jolt_field::Ring;

    let sum_long = F::from_u64(1234);
    let sum_short = F::from_u64(777);
    let mut long = DenseMember::with_sum(3, sum_long, 5);
    let mut short = DenseMember::with_sum(1, sum_short, 91);

    let mut prover_state = prover_transcript::<Blake2b512>(b"prove-batch-twin", b"");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[sum_long, sum_short], &mut prover_state);
    let coeff_long: F = prover_state.challenge().unwrap();
    let coeff_short: F = prover_state.challenge().unwrap();
    let prelude = BatchPrelude::new(
        vec![
            BatchMember {
                input_claim: sum_long,
                coefficient: coeff_long,
                rounds: 3,
                offset: 0,
            },
            BatchMember {
                input_claim: sum_short,
                coefficient: coeff_short,
                rounds: 1,
                offset: 2,
            },
        ],
        3,
        1,
    );
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    assert_eq!(
        proved.member_claims,
        vec![long.final_eval(), short.final_eval()]
    );
    let recorded = recorder
        .finish(&proved.member_claims, &mut prover_state)
        .unwrap();
    assert!(recorded.committed_witness.is_none());

    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"prove-batch-twin", b"");
    append_sumcheck_claim(&mut verifier_transcript, &sum_long);
    append_sumcheck_claim(&mut verifier_transcript, &sum_short);
    let verifier_coeff_long: F = verifier_transcript.challenge().unwrap();
    let verifier_coeff_short: F = verifier_transcript.challenge().unwrap();
    assert_eq!(
        (verifier_coeff_long, verifier_coeff_short),
        (coeff_long, coeff_short)
    );

    let claimed_sum =
        verifier_coeff_long * sum_long + verifier_coeff_short * sum_short.mul_pow_2(2);
    assert_eq!(claimed_sum, prelude.claimed_sum);
    let reduction = SumcheckProof::<F, Bn254G1>::from(recorded.proof)
        .verify_compressed_boolean(3, 1, claimed_sum, &mut verifier_transcript)
        .unwrap();
    for value in &proved.member_claims {
        verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, value);
    }

    assert_eq!(reduction.value, proved.final_claim);
    assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// Twin-transcript lock for a head-aligned member: a 1-round member at
/// `offset: 0` is active in the batch's FIRST round and then halves through
/// the trailing dummy rounds. Its kernel must emit at the dummy-round padding
/// scale (the table sums to `2^(max − rounds) · input_claim`), so its final
/// batch claim is the fully-bound value with the padding halved back out.
#[test]
fn prove_batch_clear_twin_head_aligned_member() {
    use crate::batch::{BatchMember, BatchPrelude};
    use crate::prover::{prove_batch, ProveRounds, SequentialRounds};
    use crate::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
    use crate::{append_sumcheck_claim, OPENING_CLAIM_TRANSCRIPT_LABEL};
    use jolt_field::{Field, Ring};

    let sum_long = F::from_u64(1234);
    let sum_short = F::from_u64(777);
    let mut long = DenseMember::with_sum(3, sum_long, 5);
    // The head-aligned kernel's table carries the 2^(3 - 1) padding scale.
    let mut short = DenseMember::with_sum(1, sum_short.mul_pow_2(2), 91);

    let mut prover_state = prover_transcript::<Blake2b512>(b"prove-batch-head-twin", b"");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[sum_long, sum_short], &mut prover_state);
    let coeff_long: F = prover_state.challenge().unwrap();
    let coeff_short: F = prover_state.challenge().unwrap();
    let prelude = BatchPrelude::new(
        vec![
            BatchMember {
                input_claim: sum_long,
                coefficient: coeff_long,
                rounds: 3,
                offset: 0,
            },
            BatchMember {
                input_claim: sum_short,
                coefficient: coeff_short,
                rounds: 1,
                offset: 0,
            },
        ],
        3,
        1,
    );
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    let quarter = F::from_u64(4).inverse().unwrap();
    assert_eq!(
        proved.member_claims,
        vec![long.final_eval(), short.final_eval() * quarter]
    );
    let recorded = recorder
        .finish(&proved.member_claims, &mut prover_state)
        .unwrap();

    // Verifier twin: the claimed sum is position-independent — identical to
    // the tail-aligned layout's.
    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"prove-batch-head-twin", b"");
    append_sumcheck_claim(&mut verifier_transcript, &sum_long);
    append_sumcheck_claim(&mut verifier_transcript, &sum_short);
    let verifier_coeff_long: F = verifier_transcript.challenge().unwrap();
    let verifier_coeff_short: F = verifier_transcript.challenge().unwrap();
    let claimed_sum =
        verifier_coeff_long * sum_long + verifier_coeff_short * sum_short.mul_pow_2(2);
    assert_eq!(claimed_sum, prelude.claimed_sum);
    let reduction = SumcheckProof::<F, Bn254G1>::from(recorded.proof)
        .verify_compressed_boolean(3, 1, claimed_sum, &mut verifier_transcript)
        .unwrap();
    for value in &proved.member_claims {
        verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, value);
    }

    assert_eq!(reduction.value, proved.final_claim);
    assert_eq!(reduction.point.as_slice(), proved.challenges.as_slice());
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// The committed twin of the batched engine: the same members proved through
/// the committed recorder must be byte-identical to
/// `verify_committed_consistency_dims` (coefficient draws included — no claim
/// scalars are absorbed on either side), and the retained witness must open
/// every round commitment.
#[test]
fn prove_batch_committed_twin_matches_committed_consistency() {
    use crate::batch::{BatchMember, BatchPrelude};
    use crate::prover::{prove_batch, ProveRounds, SequentialRounds};
    use crate::recorder::{CommittedSumcheckRecorder, SumcheckRecorder};

    type VC = Pedersen<Bn254G1>;
    let setup = pedersen_setup(4);

    let sum_long = F::from_u64(4242);
    let sum_short = F::from_u64(1717);
    let mut long = DenseMember::with_sum(3, sum_long, 23);
    let mut short = DenseMember::with_sum(1, sum_short, 57);

    let mut prover_state = prover_transcript::<Blake2b512>(b"prove-batch-zk-twin", b"");
    let mut recorder =
        CommittedSumcheckRecorder::<F, VC, _>::new(&setup, rand_core::OsRng).unwrap();
    recorder.absorb_input_claims(&[sum_long, sum_short], &mut prover_state);
    let coeff_long: F = prover_state.challenge().unwrap();
    let coeff_short: F = prover_state.challenge().unwrap();
    let prelude = BatchPrelude::new(
        vec![
            BatchMember {
                input_claim: sum_long,
                coefficient: coeff_long,
                rounds: 3,
                offset: 0,
            },
            BatchMember {
                input_claim: sum_short,
                coefficient: coeff_short,
                rounds: 1,
                offset: 2,
            },
        ],
        3,
        1,
    );
    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![&mut long, &mut short];
    let proved = prove_batch(
        &prelude,
        &mut members,
        &mut SequentialRounds,
        &mut recorder,
        &mut prover_state,
    )
    .unwrap();
    let recorded = recorder
        .finish(&proved.member_claims, &mut prover_state)
        .unwrap();

    // Verifier twin: the committed path absorbs no claim scalars — only the
    // coefficient squeezes and the commitments the proof carries.
    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"prove-batch-zk-twin", b"");
    let _coeff_long: F = verifier_transcript.challenge().unwrap();
    let _coeff_short: F = verifier_transcript.challenge().unwrap();
    let consistency = recorded
        .proof
        .verify_committed_consistency_dims(3, 1, &mut verifier_transcript)
        .unwrap();

    assert_eq!(consistency.challenges(), proved.challenges);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );

    let witness = recorded.committed_witness.unwrap();
    let committed = recorded.proof.as_committed().unwrap();
    assert_eq!(witness.round_coefficients.len(), committed.rounds.len());
    for ((round, coefficients), blinding) in committed
        .rounds
        .iter()
        .zip(&witness.round_coefficients)
        .zip(&witness.round_blindings)
    {
        assert!(VC::verify(
            &setup,
            &round.commitment,
            coefficients,
            blinding
        ));
    }
    for ((commitment, row), blinding) in committed
        .output_claims
        .commitments
        .iter()
        .zip(&witness.output_claim_rows)
        .zip(&witness.output_claim_blindings)
    {
        assert!(VC::verify(&setup, commitment, row, blinding));
    }
    assert_eq!(witness.output_claim_rows.concat(), proved.member_claims,);
}

/// Twin-transcript lock for the clear uni-skip prover against the verify
/// choreography `jolt-verifier`'s `uniskip::verify_clear` performs: full
/// labeled round poly, challenge, output claim absorbed under
/// `b"opening_claim"` before any later draw.
#[test]
fn prove_uniskip_clear_twin_matches_uniskip_verify() {
    use crate::prover::prove_uniskip_clear;
    use crate::{CenteredIntegerDomain, OPENING_CLAIM_TRANSCRIPT_LABEL};

    let degree = 3;
    let domain_size = 4;
    let poly = UnivariatePoly::new(vec![
        F::from_u64(3),
        F::from_u64(1),
        F::from_u64(4),
        F::from_u64(1),
    ]);
    let domain = CenteredIntegerDomain::new(domain_size);
    let coefficients = domain.round_sum_coefficients(degree).unwrap();
    let input_claim =
        <UnivariatePoly<F> as ClearRound<F>>::coefficient_linear_combination(&poly, &coefficients);

    let mut prover_state = prover_transcript::<Blake2b512>(b"uniskip-twin", b"");
    let proved =
        prove_uniskip_clear::<F, _>(poly, input_claim, degree, domain_size, &mut prover_state)
            .unwrap();

    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"uniskip-twin", b"");
    let reduction = SumcheckProof::<F, Bn254G1>::from(proved.proof)
        .verify(
            &SumcheckClaim::new(1, degree, input_claim),
            domain,
            crate::UNISKIP_ROUND_TRANSCRIPT_LABEL,
            &mut verifier_transcript,
        )
        .unwrap();
    assert_eq!(reduction.value, proved.output_claim);
    verifier_transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &proved.output_claim);

    assert_eq!(reduction.point.as_slice(), &[proved.challenge]);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
}

/// The committed uni-skip twin: `prove_uniskip_committed` must be
/// byte-identical to `verify_committed_consistency` (the transcript path of
/// `uniskip::verify_zk` — its commitment-count check is transcript-pure), and
/// the witness must open the round and output-claim commitments.
#[test]
fn prove_uniskip_committed_twin_matches_committed_consistency() {
    use crate::prover::prove_uniskip_committed;
    use crate::CenteredIntegerDomain;

    type VC = Pedersen<Bn254G1>;
    let setup = pedersen_setup(4);

    let degree = 3;
    let domain_size = 4;
    let poly = UnivariatePoly::new(vec![
        F::from_u64(2),
        F::from_u64(7),
        F::from_u64(1),
        F::from_u64(8),
    ]);
    let domain = CenteredIntegerDomain::new(domain_size);
    let coefficients = domain.round_sum_coefficients(degree).unwrap();
    let input_claim =
        <UnivariatePoly<F> as ClearRound<F>>::coefficient_linear_combination(&poly, &coefficients);

    let mut prover_state = prover_transcript::<Blake2b512>(b"uniskip-zk-twin", b"");
    let proved = prove_uniskip_committed::<F, VC, _, _>(
        poly,
        input_claim,
        degree,
        domain_size,
        &setup,
        rand_core::OsRng,
        &mut prover_state,
    )
    .unwrap();

    let mut verifier_transcript = prover_transcript::<Blake2b512>(b"uniskip-zk-twin", b"");
    let consistency = proved
        .proof
        .verify_committed_consistency(SumcheckStatement::new(1, degree), &mut verifier_transcript)
        .unwrap();

    assert_eq!(consistency.challenges(), vec![proved.challenge]);
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
    assert_eq!(
        proved.witness.output_claim_rows.concat(),
        vec![proved.output_claim],
    );
}

#[test]
fn bn254_round_proofs_carry_no_nonce_bytes() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let proof = honest_prove(&evals, 3, &mut transcript);
    assert!(proof.pow_nonces.is_empty());

    let config = bincode::config::standard();
    let wire = bincode::serde::encode_to_vec(&proof, config).unwrap();
    let rounds_only = bincode::serde::encode_to_vec(&proof.round_polynomials, config).unwrap();
    // The header 2·len replaces the length prefix in the same single byte.
    assert_eq!((wire[0], rounds_only[0]), (6, 3));
    assert_eq!(wire[1..], rounds_only[1..]);
    let (decoded, read): (ClearSumcheckProof<F>, usize) =
        bincode::serde::decode_from_slice(&wire, config).unwrap();
    assert_eq!((decoded, read), (proof, wire.len()));
}

#[test]
fn nonced_round_trips_and_rejects_malformed_input() {
    let config = bincode::config::standard();
    let decode = |wire: &[u8]| bincode::serde::decode_from_slice::<Nonced<u64>, _>(wire, config);
    for (messages, pow_nonces) in [
        (vec![], vec![]),
        (vec![1, 2, 3], vec![]),
        (vec![1, 2, 3], vec![4, 5, 6]),
    ] {
        let wire = bincode::serde::encode_to_vec(
            NoncedRef {
                messages: &messages,
                pow_nonces: &pow_nonces,
            },
            config,
        )
        .unwrap();
        let expected = Nonced {
            messages,
            pow_nonces,
        };
        assert_eq!(decode(&wire).unwrap(), (expected, wire.len()));
        for cut in 0..wire.len() {
            assert!(decode(&wire[..cut]).is_err(), "truncated at {cut}");
        }
    }

    // One message, the nonce flag set, then an empty nonce list.
    assert!(decode(&[3, 7, 0]).is_err());
}

/// Replays rounds `0..round` with their honest nonces on a transcript for
/// `session`, absorbs `rounds[round]`, and returns the first nonce that fails
/// that round's proof of work.
fn failing_nonce<R: RoundMessage>(
    rounds: &[R],
    nonces: &[u32],
    round: usize,
    degree: usize,
    session: &[u8],
) -> u32 {
    let bits = round_pow_bits::<P, PT>(degree).unwrap();
    let probe = || {
        let mut transcript = prover_transcript::<Blake2b512>(session, b"");
        for (message, &nonce) in rounds[..round].iter().zip(nonces) {
            message.append_to_transcript(&mut transcript);
            transcript.check_public_grind(bits, Some(nonce)).unwrap();
            let _challenge: P = transcript.challenge().unwrap();
        }
        rounds[round].append_to_transcript(&mut transcript);
        transcript
    };
    (0..u32::MAX)
        .find(|&nonce| probe().check_public_grind(bits, Some(nonce)).is_err())
        .unwrap()
}

#[test]
fn clear_verify_rejects_wrong_pow_nonce() {
    let evals: Vec<P> = (1..=8).map(P::from_u64).collect();
    let claim = SumcheckClaim::new(3, 1, compute_sum(&evals));
    let mut prover_state = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let mut proof = honest_prove(&evals, 3, &mut prover_state);
    proof.pow_nonces[1] = failing_nonce(
        &proof.round_polynomials,
        &proof.pow_nonces,
        1,
        1,
        b"sumcheck-test",
    );

    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let result = SumcheckVerifier::verify(
        &claim,
        &proof.round_polynomials,
        &proof.pow_nonces,
        BooleanHypercube,
        &mut transcript,
    );

    assert!(matches!(
        result,
        Err(SumcheckError::ProofOfWorkRejected { round: 1 })
    ));
}

#[test]
fn clear_verify_rejects_dropped_and_extra_pow_nonces() {
    let evals: Vec<P> = (1..=8).map(P::from_u64).collect();
    let claim = SumcheckClaim::new(3, 1, compute_sum(&evals));
    let mut prover_state = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let proof = honest_prove(&evals, 3, &mut prover_state);
    let dropped = &proof.pow_nonces[..2];
    let extra = [proof.pow_nonces.as_slice(), &[0]].concat();

    for (nonces, got) in [(dropped, 2), (extra.as_slice(), 4)] {
        let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
        let mut untouched = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof.round_polynomials,
            nonces,
            BooleanHypercube,
            &mut transcript,
        );

        assert!(matches!(
            result,
            Err(SumcheckError::WrongNumberOfPowNonces { expected: 3, got: actual }) if actual == got
        ));
        assert_eq!(
            transcript.verifier_message::<[u8; 32]>(),
            untouched.verifier_message::<[u8; 32]>()
        );
    }
}

#[test]
fn compressed_verify_rejects_wrong_and_dropped_pow_nonces() {
    let evals: Vec<P> = (1..=8).map(P::from_u64).collect();
    let claim = SumcheckClaim::new(3, 1, compute_sum(&evals));
    let label = SUMCHECK_ROUND_TRANSCRIPT_LABEL;
    let mut prover_state = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    let (clear_proof, compressed_proof) =
        honest_prove_compressed_labeled(&evals, 3, label, &mut prover_state);
    let wrapped = clear_proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, label))
        .collect::<Vec<_>>();

    let mut wrong = compressed_proof.clone();
    wrong.pow_nonces[2] = failing_nonce(
        &wrapped,
        &compressed_proof.pow_nonces,
        2,
        1,
        b"sumcheck-test",
    );
    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    assert!(matches!(
        wrong.verify(&claim, BooleanHypercube, label, &mut transcript),
        Err(SumcheckError::ProofOfWorkRejected { round: 2 })
    ));

    let mut dropped = compressed_proof;
    dropped.pow_nonces.clear();
    let mut transcript = prover_transcript::<Blake2b512>(b"sumcheck-test", b"");
    assert!(matches!(
        dropped.verify(&claim, BooleanHypercube, label, &mut transcript),
        Err(SumcheckError::WrongNumberOfPowNonces {
            expected: 3,
            got: 0
        })
    ));
}

#[test]
fn committed_rounds_reject_wrong_and_dropped_pow_nonces() {
    let rounds = vec![
        CommittedRound {
            commitment: P::from_u64(11),
            degree: 2,
        },
        CommittedRound {
            commitment: P::from_u64(12),
            degree: 2,
        },
    ];
    let statement = SumcheckStatement::new(2, 2);
    let mut prover = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    let mut pow_nonces = Vec::new();
    for round in &rounds {
        prover.append(&LabelWithCount(b"sumcheck_commitment", round.degree as u64));
        prover.append(&round.commitment);
        pow_nonces.extend(
            prover
                .public_grind(round_pow_bits::<P, PT>(2).unwrap())
                .unwrap(),
        );
        let _challenge: P = prover.challenge().unwrap();
    }

    let bits = round_pow_bits::<P, PT>(2).unwrap();
    let mut wrong = pow_nonces.clone();
    wrong[0] = (0..u32::MAX)
        .find(|&nonce| {
            let mut probe = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
            probe.append(&LabelWithCount(b"sumcheck_commitment", 2));
            probe.append(&rounds[0].commitment);
            probe.check_public_grind(bits, Some(nonce)).is_err()
        })
        .unwrap();
    let mut transcript = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    assert!(matches!(
        SumcheckVerifier::verify_committed_round_consistency::<P, _, _>(
            statement,
            &rounds,
            &wrong,
            &mut transcript,
        ),
        Err(SumcheckError::ProofOfWorkRejected { round: 0 })
    ));

    let mut transcript = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    assert!(matches!(
        SumcheckVerifier::verify_committed_round_consistency::<P, _, _>(
            statement,
            &rounds,
            &pow_nonces[..1],
            &mut transcript,
        ),
        Err(SumcheckError::WrongNumberOfPowNonces {
            expected: 2,
            got: 1
        })
    ));

    let mut transcript = prover_transcript::<Blake2b512>(b"committed-sumcheck", b"");
    assert!(
        SumcheckVerifier::verify_committed_round_consistency::<P, _, _>(
            statement,
            &rounds,
            &pow_nonces,
            &mut transcript,
        )
        .is_ok()
    );
}
