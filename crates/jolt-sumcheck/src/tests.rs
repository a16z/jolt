#![expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::wildcard_enum_match_arm,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures, index directly, and match errors with catch-all arms"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Field, Fr, Prime128OffsetA7F7, Prime32Offset99, Ring};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_transcript::{
    prover_transcript, Blake2b512, Encoding, FieldTranscript, ProverTranscript, Transcript,
};

use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckStatement};
use crate::committed::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, CommittedSumcheckConsistency,
    CommittedSumcheckProof, VerifiedCommittedRound,
};
use crate::error::SumcheckError;
use crate::proof::{ClearProof, ClearSumcheckProof, CompressedSumcheckProof, SumcheckProof};
use crate::round_proof::{append_compressed_round, ClearRound, CompressedRoundPoly, RoundMessage};
use crate::verifier::SumcheckVerifier;
use crate::{
    BatchedCommittedSumcheckConsistency, BooleanHypercube, CenteredIntegerDomain, SumcheckDomain,
};

type F = Fr;
/// A challenge set that would need proof of work, which committed proofs
/// refuse.
type P = Prime128OffsetA7F7;

fn honest_prove<E: Field + Encoding, T: FieldTranscript<E> + ProverTranscript>(
    evals: &[E],
    num_vars: usize,
    transcript: &mut T,
) -> ClearSumcheckProof<E> {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);

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

        // Absorb through the same path the verifier uses.
        <UnivariatePoly<E> as RoundMessage>::append_to_transcript(&round_poly, transcript);

        let r: E = transcript.challenge().unwrap();
        round_polys.push(round_poly);

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }

    ClearSumcheckProof {
        round_polynomials: round_polys,
    }
}

fn honest_prove_compressed<E: Field + Encoding, T: FieldTranscript<E> + ProverTranscript>(
    evals: &[E],
    num_vars: usize,
    transcript: &mut T,
) -> (ClearSumcheckProof<E>, CompressedSumcheckProof<E>) {
    let mut buf = evals.to_vec();
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut compressed_round_polys = Vec::with_capacity(num_vars);

    for _round in 0..num_vars {
        let half = buf.len() / 2;

        let mut eval_0 = E::from_u64(0);
        let mut eval_1 = E::from_u64(0);
        for i in 0..half {
            eval_0 += buf[i];
            eval_1 += buf[i + half];
        }

        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);
        let compressed = CompressedRoundPoly::new(&round_poly);
        <CompressedRoundPoly<'_, E> as RoundMessage>::append_to_transcript(&compressed, transcript);

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
        },
        CompressedSumcheckProof {
            round_polynomials: compressed_round_polys,
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

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let proof = honest_prove(&evals, num_vars, &mut prover_state);

    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
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
fn centered_integer_domain_verifies_round_sum() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 2,
        claimed_sum: F::from_u64(16),
    };

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-integer-domain-test"),
        b"",
    );
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

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-integer-domain-test"),
        b"",
    );
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

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-integer-domain-test"),
        b"",
    );
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

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-integer-domain-test"),
        b"",
    );
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

    let mut pt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let mut proof = honest_prove(&evals, 3, &mut pt);

    let bad_coeffs = vec![F::from_u64(999), F::from_u64(1)];
    proof.round_polynomials[0] = UnivariatePoly::new(bad_coeffs);

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
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

    let mut pt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let mut proof = honest_prove(&evals, 3, &mut pt);

    let _ = proof.round_polynomials.pop();

    let claim = SumcheckClaim {
        num_vars: 3,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
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

    let mut pt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let mut proof = honest_prove(&evals, 2, &mut pt);

    proof.round_polynomials[0] =
        UnivariatePoly::new(vec![sum, F::from_u64(1), F::from_u64(0), F::from_u64(1)]);

    let claim = SumcheckClaim {
        num_vars: 2,
        degree: 1,
        claimed_sum: sum,
    };

    let mut vt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
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
fn clear_round_absorbs_count_word_then_coefficients() {
    let poly = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);

    let mut t1 =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    <UnivariatePoly<F> as RoundMessage>::append_to_transcript(&poly, &mut t1);
    let c1: F = t1.challenge().unwrap();

    let mut t2 =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    t2.append(&2u64);
    for coeff in poly.coefficients() {
        t2.append(&coeff);
    }
    let c2: F = t2.challenge().unwrap();

    assert_eq!(
        c1, c2,
        "a round must absorb its coefficient count, then its coefficients"
    );
}

#[test]
fn clear_round_binds_the_coefficient_count() {
    let short = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3)]);
    let long = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(3), F::from_u64(0)]);
    let draw = |poly: &UnivariatePoly<F>| -> F {
        let mut transcript =
            prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
        <UnivariatePoly<F> as RoundMessage>::append_to_transcript(poly, &mut transcript);
        transcript.challenge().unwrap()
    };

    assert_ne!(draw(&short), draw(&long));
}

#[test]
fn clear_round_verifier_compressed_matches_manual_absorption() {
    let poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let compressed = CompressedRoundPoly::new(&poly);

    let mut t1 =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    <CompressedRoundPoly<'_, F> as RoundMessage>::append_to_transcript(&compressed, &mut t1);
    let ch1: F = t1.challenge().unwrap();

    // Manual absorb matching the compressed wire format: count word d, c0, c2..cd.
    let mut t2 =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let coeffs = poly.coefficients();
    t2.append(&((coeffs.len() - 1) as u64));
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

/// The prover's compressed round and the verifier's compressed-proof loop
/// absorb alike: one count word, then `[c0, c2, c3]` as one message
/// zero-padded to a word. For 4-byte coefficients that is 12 bytes and 4
/// zeros, so the next message starts on a word boundary; for BN254 there is
/// no padding.
fn compressed_round_absorbs_like_the_verifier<E: Field + Encoding>() {
    let poly = UnivariatePoly::new((1..=4).map(|i| E::from_u64(i * 1000 + 7)).collect());
    let coeffs = poly.coefficients();
    let mut layout = 3u64.to_le_bytes().to_vec();
    for c in [coeffs[0], coeffs[2], coeffs[3]] {
        layout.extend_from_slice(c.encode().as_ref());
    }
    layout.resize(layout.len().next_multiple_of(8), 0);
    let next = 0x0123_4567_89ab_cdef_u64;
    let transcript =
        || prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");

    let mut prover = transcript();
    <CompressedRoundPoly<'_, E> as RoundMessage>::append_to_transcript(
        &CompressedRoundPoly::new(&poly),
        &mut prover,
    );
    prover.append(&next);
    let mut verifier = transcript();
    append_compressed_round(&mut verifier, poly.compress().coeffs_except_linear_term());
    verifier.append(&next);
    let mut expected = transcript();
    for byte in &layout {
        expected.public_message(byte);
    }
    expected.public_message(&next);

    let challenge = prover.verifier_message::<[u8; 32]>();
    assert_eq!(challenge, verifier.verifier_message::<[u8; 32]>());
    assert_eq!(challenge, expected.verifier_message::<[u8; 32]>());
}

#[test]
fn compressed_round_absorbs_like_the_verifier_for_four_byte_and_bn254_fields() {
    assert_eq!(Prime32Offset99::from_u64(1).encode().as_ref().len(), 4);
    compressed_round_absorbs_like_the_verifier::<Prime32Offset99>();
    compressed_round_absorbs_like_the_verifier::<F>();
}

#[test]
fn clear_round_verifier_compressed_rejects_wrong_running_sum() {
    // Even when the linear term is omitted from absorption, the sum check
    // (s(0) + s(1) == running_sum) still binds every coefficient.
    let poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]);
    let wrong_running_sum = F::from_u64(999);
    let compressed = CompressedRoundPoly::new(&poly);

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
    let compressed = CompressedRoundPoly::new(&degree_zero);

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

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let (clear_proof, compressed_proof) =
        honest_prove_compressed(&evals, num_vars, &mut prover_state);
    let claim = SumcheckClaim {
        num_vars,
        degree: 1,
        claimed_sum: sum,
    };

    let wrapped = clear_proof
        .round_polynomials
        .iter()
        .map(CompressedRoundPoly::new)
        .collect::<Vec<_>>();
    let mut borrowed_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let borrowed =
        SumcheckVerifier::verify(&claim, &wrapped, BooleanHypercube, &mut borrowed_transcript)
            .unwrap();

    let mut owned_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let owned = compressed_proof
        .verify(&claim, BooleanHypercube, &mut owned_transcript)
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
    };
    let claim = SumcheckClaim {
        num_vars: 1,
        degree: 1,
        claimed_sum: F::from_u64(0),
    };

    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let result = proof.verify(&claim, BooleanHypercube, &mut transcript);

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

    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let result = proof.verify(&claim, BooleanHypercube, &mut transcript);

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

    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
    let result = proof.verify(&claim, BooleanHypercube, &mut transcript);

    assert!(matches!(
        result,
        Err(SumcheckError::CompressedPolynomialTooShort { round: 0, got: 0 })
    ));
}

#[test]
fn sumcheck_proof_verify_dispatches_full_clear() {
    let round_poly = UnivariatePoly::new(vec![F::from_u64(2), F::from_u64(3)]);
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Full(ClearSumcheckProof {
        round_polynomials: vec![round_poly],
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(7));

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-proof-dispatch"),
        b"",
    );
    let reduction = proof
        .verify(&claim, BooleanHypercube, &mut transcript)
        .unwrap();

    assert_eq!(reduction.point.len(), 1);
}

#[test]
fn sumcheck_proof_verify_rejects_wrong_clear_encoding() {
    let proof = SumcheckProof::<F, F>::Clear(ClearProof::Compressed(CompressedSumcheckProof {
        round_polynomials: Vec::new(),
    }));
    let claim = SumcheckClaim::new(1, 1, F::from_u64(0));

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-proof-dispatch"),
        b"",
    );
    let result = proof.verify(&claim, BooleanHypercube, &mut transcript);

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

    let mut transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"sumcheck-proof-dispatch"),
        b"",
    );
    let result = proof.verify(&claim, BooleanHypercube, &mut transcript);

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

    let mut manual =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-sumcheck"), b"");
    let mut expected_challenges: Vec<F> = Vec::new();
    for round in &rounds {
        manual.append(&(round.degree as u64));
        manual.append(&round.commitment);
        expected_challenges.push(manual.challenge().unwrap());
    }

    let mut verifier =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-sumcheck"), b"");
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-sumcheck"), b"");

    let result = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-sumcheck"), b"");
    let mut untouched =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-sumcheck"), b"");

    let result = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(1, 2),
        &rounds,
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
fn committed_output_claims_absorb_commitments_in_order() {
    let output_claims = CommittedOutputClaims {
        commitments: vec![F::from_u64(3), F::from_u64(5), F::from_u64(8)],
    };

    let mut actual =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-output"), b"");
    actual.append(&output_claims);

    let mut expected =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-output"), b"");
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

    let mut manual =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-proof"), b"");
    let mut expected_challenges: Vec<F> = Vec::new();
    for round in &proof.rounds {
        manual.append(&(round.degree as u64));
        manual.append(&round.commitment);
        expected_challenges.push(manual.challenge().unwrap());
    }
    for commitment in &proof.output_claims.commitments {
        manual.append(&commitment);
    }

    let mut verifier =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-proof"), b"");
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

    let mut vt =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"sumcheck-test"), b"");
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

    let num_vars = 3;
    let evals: Vec<F> = (0..1u64 << num_vars)
        .map(|i| F::from_u64(3 * i + 7))
        .collect();
    let claimed_sum = compute_sum(&evals);

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"recorder-test"), b"");
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    recorder.absorb_input_claims(&[claimed_sum], &mut prover_state);

    let mut buf = evals;
    for _round in 0..num_vars {
        let half = buf.len() / 2;
        let eval_0: F = buf[..half].iter().copied().sum();
        let eval_1: F = buf[half..].iter().copied().sum();
        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);

        let r = recorder
            .absorb_round(&round_poly, &mut prover_state)
            .unwrap();

        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }
    let final_eval = buf[0];
    let recorded = recorder.finish(&[final_eval], &mut prover_state).unwrap();

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"recorder-test"), b"");
    verifier_transcript.append(&claimed_sum);
    let reduction = SumcheckProof::<F, Bn254G1>::from(recorded.proof)
        .verify_compressed_boolean(num_vars, 1, claimed_sum, &mut verifier_transcript)
        .unwrap();
    verifier_transcript.append(&reduction.value);

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
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"zero-degree-batch"), b"");
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
    use jolt_field::Ring;

    let sum_long = F::from_u64(1234);
    let sum_short = F::from_u64(777);
    let mut long = DenseMember::with_sum(3, sum_long, 5);
    let mut short = DenseMember::with_sum(1, sum_short, 91);

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"prove-batch-twin"), b"");
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

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"prove-batch-twin"), b"");
    verifier_transcript.append(&sum_long);
    verifier_transcript.append(&sum_short);
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
        verifier_transcript.append(value);
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
    use jolt_field::{Field, Ring};

    let sum_long = F::from_u64(1234);
    let sum_short = F::from_u64(777);
    let mut long = DenseMember::with_sum(3, sum_long, 5);
    // The head-aligned kernel's table carries the 2^(3 - 1) padding scale.
    let mut short = DenseMember::with_sum(1, sum_short.mul_pow_2(2), 91);

    let mut prover_state = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"prove-batch-head-twin"),
        b"",
    );
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
    let mut verifier_transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"prove-batch-head-twin"),
        b"",
    );
    verifier_transcript.append(&sum_long);
    verifier_transcript.append(&sum_short);
    let verifier_coeff_long: F = verifier_transcript.challenge().unwrap();
    let verifier_coeff_short: F = verifier_transcript.challenge().unwrap();
    let claimed_sum =
        verifier_coeff_long * sum_long + verifier_coeff_short * sum_short.mul_pow_2(2);
    assert_eq!(claimed_sum, prelude.claimed_sum);
    let reduction = SumcheckProof::<F, Bn254G1>::from(recorded.proof)
        .verify_compressed_boolean(3, 1, claimed_sum, &mut verifier_transcript)
        .unwrap();
    for value in &proved.member_claims {
        verifier_transcript.append(value);
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

    let mut prover_state = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"prove-batch-zk-twin"),
        b"",
    );
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
    let mut verifier_transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"prove-batch-zk-twin"),
        b"",
    );
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
/// round poly with its count word, challenge, then the output claim before
/// any later draw.
#[test]
fn prove_uniskip_clear_twin_matches_uniskip_verify() {
    use crate::prover::prove_uniskip_clear;
    use crate::CenteredIntegerDomain;

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

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"uniskip-twin"), b"");
    let proved =
        prove_uniskip_clear::<F, _>(poly, input_claim, degree, domain_size, &mut prover_state)
            .unwrap();

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"uniskip-twin"), b"");
    let reduction = SumcheckProof::<F, Bn254G1>::from(proved.proof)
        .verify(
            &SumcheckClaim::new(1, degree, input_claim),
            domain,
            &mut verifier_transcript,
        )
        .unwrap();
    assert_eq!(reduction.value, proved.output_claim);
    verifier_transcript.append(&proved.output_claim);

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

    let mut prover_state =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"uniskip-zk-twin"), b"");
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

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"uniskip-zk-twin"), b"");
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

/// Committed proofs carry no nonces, so the committed path refuses a
/// challenge set whose rounds would need proof of work.
#[test]
fn committed_rounds_reject_a_challenge_set_that_needs_pow() {
    let rounds = vec![CommittedRound {
        commitment: P::from_u64(11),
        degree: 2,
    }];
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(b"committed-sumcheck"), b"");
    assert!(matches!(
        SumcheckVerifier::verify_committed_round_consistency::<P, _, _>(
            SumcheckStatement::new(1, 2),
            &rounds,
            &mut transcript,
        ),
        Err(SumcheckError::NeedsProofOfWork)
    ));
}
