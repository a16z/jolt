#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{AkitaNativeBatching, AkitaProverHint, AkitaScheme};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, VerifierOpeningClaim,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{
    batch_polynomials, f, layout, native_setup, native_statement, polynomial, setup_for,
    single_statement,
};

#[test]
fn akita_native_batching_roundtrips_grouped_commitment() {
    let (prover_setup, verifier_setup) = native_setup();
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let statement = native_statement(commitment, &point, [eval_a, eval_b]);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-bb-roundtrip");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut prover_transcript,
    )
    .expect("black-box proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-bb-roundtrip");
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("black-box proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn akita_native_batching_rejects_malformed_statements() {
    let (prover_setup, _) = native_setup();
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (group_commitment, group_hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let (other_commitment, _) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[polynomial(13, 80)])
            .expect("other commit should succeed");

    let mut transcript = Blake2bTranscript::new(b"akita-bb-empty");
    assert!(matches!(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            Vec::new(),
            Vec::new(),
            AkitaProverHint::default(),
            &mut transcript,
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));

    let mixed_commitments = vec![
        VerifierOpeningClaim {
            commitment: group_commitment.clone(),
            evaluation: EvaluationClaim::new(point.clone(), eval_a),
        },
        VerifierOpeningClaim {
            commitment: other_commitment,
            evaluation: EvaluationClaim::new(point.clone(), eval_b),
        },
    ];
    let mut transcript = Blake2bTranscript::new(b"akita-bb-mixed-commit");
    assert!(matches!(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            mixed_commitments,
            batch_polynomials([&poly_a, &poly_b]),
            group_hint.clone(),
            &mut transcript,
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));

    let mut mixed_points = native_statement(group_commitment.clone(), &point, [eval_a, eval_b]);
    let mut shifted_point = point.clone();
    shifted_point[0] += f(1);
    mixed_points[1].evaluation = EvaluationClaim::new(shifted_point, eval_b);
    let mut transcript = Blake2bTranscript::new(b"akita-bb-mixed-points");
    assert!(matches!(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            mixed_points,
            batch_polynomials([&poly_a, &poly_b]),
            group_hint.clone(),
            &mut transcript,
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));

    let one_claim_for_two_slots = single_statement(group_commitment, &point, eval_a);
    let mut transcript = Blake2bTranscript::new(b"akita-bb-claim-count");
    assert!(matches!(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            one_claim_for_two_slots,
            batch_polynomials([&poly_a, &poly_b]),
            group_hint,
            &mut transcript,
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));
}

#[test]
fn akita_native_batching_rejects_bad_prover_witnesses() {
    let (prover_setup, _) = native_setup();
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let (_, other_hint) = AkitaScheme::commit_group(
        &prover_setup,
        layout(7),
        &[polynomial(13, 80), polynomial(13, 100)],
    )
    .expect("other grouped commit should succeed");
    let statement = native_statement(commitment, &point, [eval_a, eval_b]);

    let mut transcript = Blake2bTranscript::new(b"akita-bb-wrong-hint");
    assert!(
        matches!(
            <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                statement.clone(),
                batch_polynomials([&poly_a, &poly_b]),
            other_hint,
                &mut transcript,
            ),
            Err(OpeningsError::InvalidBatch(message)) if message.contains("hint")
        ),
        "mismatched prover hint should reject"
    );

    let mut transcript = Blake2bTranscript::new(b"akita-bb-wrong-count");
    assert!(matches!(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            batch_polynomials([&poly_a]),
            hint.clone(),
            &mut transcript,
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));

    let wrong_dimension = polynomial(12, 200);
    let mut transcript = Blake2bTranscript::new(b"akita-bb-wrong-dim");
    assert!(matches!(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement,
            batch_polynomials([&poly_a, &wrong_dimension]),
            hint,
            &mut transcript,
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));
}

#[test]
fn akita_native_batching_rejects_tampered_verifier_inputs() {
    let (prover_setup, verifier_setup) = native_setup();
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let statement = native_statement(commitment.clone(), &point, [eval_a, eval_b]);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-bb-tamper");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut prover_transcript,
    )
    .expect("black-box proof should be produced");

    let mut tampered_value = statement.clone();
    tampered_value[0].evaluation.value += f(1);
    assert_native_verify_rejects(&verifier_setup, tampered_value, &proof);

    let mut tampered_point = statement.clone();
    let mut shifted_point = point.clone();
    shifted_point[1] += f(1);
    tampered_point[1].evaluation = EvaluationClaim::new(shifted_point, eval_b);
    assert_native_verify_rejects(&verifier_setup, tampered_point, &proof);

    let (other_commitment, _) = AkitaScheme::commit_group(
        &prover_setup,
        layout(7),
        &[polynomial(13, 80), polynomial(13, 100)],
    )
    .expect("other grouped commit should succeed");
    let tampered_commitment = native_statement(other_commitment, &point, [eval_a, eval_b]);
    assert_native_verify_rejects(&verifier_setup, tampered_commitment, &proof);

    let (_, wrong_layout_setup) = setup_for(13, 2, layout(8));
    assert_native_verify_rejects(&wrong_layout_setup, statement, &proof);
}

fn assert_native_verify_rejects(
    setup: &<AkitaScheme as CommitmentScheme>::VerifierSetup,
    statement: jolt_akita::AkitaNativeBatchStatement,
    proof: &jolt_akita::AkitaBatchProof,
) {
    let mut transcript = Blake2bTranscript::new(b"akita-bb-tamper");
    assert!(<AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        setup,
        &statement,
        proof,
        &mut transcript,
    )
    .is_err());
}

fn expect_invalid_batch<T: std::fmt::Debug>(
    result: Result<T, OpeningsError>,
    expected_fragment: &str,
) {
    let err = result.expect_err("statement must be rejected");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains(expected_fragment)),
        "expected InvalidBatch containing {expected_fragment:?}, got: {err}"
    );
}

#[test]
fn akita_native_batching_rejects_point_commitment_dimension_mismatch() {
    let (prover_setup, _) = native_setup();
    let poly = polynomial(13, 1);
    let short_point: Vec<_> = (0..12).map(|index| f(index + 2)).collect();
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
            .expect("commit should succeed");
    let statement = single_statement(commitment, &short_point, f(9));

    let mut transcript = Blake2bTranscript::new(b"akita-bb-short-point");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement,
            batch_polynomials([&poly]),
            hint,
            &mut transcript,
        ),
        "12 variables but commitment has 13",
    );
}

/// The exact-dimension and group-width checks run against the verifier's own
/// setup, so a statement built for one setup must reject against another.
#[test]
fn akita_native_batching_rejects_statements_outside_the_verifier_setup() {
    let (small_setup, _) = setup_for(13, 1, layout(7));
    let small_poly = polynomial(13, 1);
    let small_point: Vec<_> = (0..13).map(|index| f(index + 2)).collect();
    let small_eval = small_poly.evaluate(&small_point);
    let (small_commitment, small_hint) =
        AkitaScheme::commit_group(&small_setup, layout(7), std::slice::from_ref(&small_poly))
            .expect("commit should succeed");
    let small_statement = single_statement(small_commitment, &small_point, small_eval);
    let mut transcript = Blake2bTranscript::new(b"akita-bb-cross-setup");
    let small_proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &small_setup,
        small_statement.clone(),
        batch_polynomials([&small_poly]),
        small_hint,
        &mut transcript,
    )
    .expect("proof should be produced");

    // A 13-variable commitment against a 14-variable verifier setup.
    let (_, wider_verifier) = setup_for(14, 2, layout(7));
    let mut transcript = Blake2bTranscript::new(b"akita-bb-cross-setup");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &wider_verifier,
            &small_statement,
            &small_proof,
            &mut transcript,
        ),
        "does not match exact setup dimension",
    );

    // A two-polynomial group against a verifier setup capped at one slot.
    let (two_slot_setup, _) = native_setup();
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let point: Vec<_> = (0..13).map(|index| f(index + 2)).collect();
    let (group_commitment, group_hint) = AkitaScheme::commit_group(
        &two_slot_setup,
        layout(7),
        &[poly_a.clone(), poly_b.clone()],
    )
    .expect("group commit should succeed");
    let group_statement = native_statement(
        group_commitment,
        &point,
        [poly_a.evaluate(&point), poly_b.evaluate(&point)],
    );
    let mut transcript = Blake2bTranscript::new(b"akita-bb-cross-setup");
    let group_proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &two_slot_setup,
        group_statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        group_hint,
        &mut transcript,
    )
    .expect("group proof should be produced");
    let (_, one_slot_verifier) = setup_for(13, 1, layout(7));
    let mut transcript = Blake2bTranscript::new(b"akita-bb-cross-setup");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &one_slot_verifier,
            &group_statement,
            &group_proof,
            &mut transcript,
        ),
        "but setup supports 1",
    );
}

/// A dense-flavor commitment claiming a one-hot chunk size is internally
/// inconsistent and must be rejected before any backend work.
#[test]
fn akita_native_batching_rejects_dense_commitment_with_chunk_size() {
    let (prover_setup, verifier_setup) = native_setup();
    let poly = polynomial(13, 1);
    let point: Vec<_> = (0..13).map(|index| f(index + 2)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
            .expect("commit should succeed");
    let statement = single_statement(commitment.clone(), &point, eval);
    let mut transcript = Blake2bTranscript::new(b"akita-bb-full-chunk");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly]),
        hint,
        &mut transcript,
    )
    .expect("proof should be produced");

    let mut forged = serde_json::to_value(&commitment).expect("commitment serializes");
    *forged
        .get_mut("one_hot_k")
        .expect("commitment exposes one_hot_k") = serde_json::json!(4);
    let forged: jolt_akita::AkitaCommitment =
        serde_json::from_value(forged).expect("forged commitment deserializes");
    let forged_statement = single_statement(forged, &point, eval);

    let mut transcript = Blake2bTranscript::new(b"akita-bb-full-chunk");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &forged_statement,
            &proof,
            &mut transcript,
        ),
        "must not declare a one-hot chunk size",
    );
}

/// One-hot and sparse-unit hints certify that the committed data was
/// one-hot; handing the prover dense witnesses for such a hint must reject.
#[test]
fn akita_native_batching_rejects_dense_witnesses_for_one_hot_hints() {
    use jolt_akita::{AkitaSetupParams, AKITA_ONE_HOT_K16};
    use jolt_poly::OneHotPolynomial;

    let (one_hot_setup, _) = AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        12,
        1,
        layout(7),
        AKITA_ONE_HOT_K16,
    ))
    .expect("one-hot setup should build");
    let one_hot_indices: Vec<_> = (0..256usize)
        .map(|row| {
            if row % 6 == 5 {
                None
            } else {
                Some(((row * 5) % 16) as u8)
            }
        })
        .collect();
    let one_hot = OneHotPolynomial::new(AKITA_ONE_HOT_K16, one_hot_indices);
    let (commitment, hint) = AkitaScheme::commit_one_hot_group(
        &one_hot_setup,
        layout(7),
        std::slice::from_ref(&one_hot),
    )
    .expect("one-hot commit should succeed");
    let dense_12 = polynomial(12, 1);
    let point_12: Vec<_> = (0..12).map(|index| f(index as u64 + 2)).collect();
    let statement = single_statement(commitment, &point_12, dense_12.evaluate(&point_12));
    let mut transcript = Blake2bTranscript::new(b"akita-bb-dense-for-onehot");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &one_hot_setup,
            statement,
            batch_polynomials([&dense_12]),
            hint,
            &mut transcript,
        ),
        "one_hot prover hint requires one-hot witness polynomials",
    );

    // Same guard for the sparse-unit representation of a public one-hot
    // commitment (K=4 rides the dense-flavor sparse path).
    let (sparse_setup, _) = setup_for(13, 1, layout(7));
    let sparse_indices: Vec<_> = (0..(1usize << 13) / 4)
        .map(|row| {
            if row % 5 == 4 {
                None
            } else {
                Some((row % 4) as u8)
            }
        })
        .collect();
    let sparse_source = OneHotPolynomial::new(4, sparse_indices);
    let (sparse_commitment, sparse_hint) =
        AkitaScheme::commit(&sparse_source, &sparse_setup).expect("sparse commit should succeed");
    let dense_13 = polynomial(13, 1);
    let point_13: Vec<_> = (0..13).map(|index| f(index as u64 + 2)).collect();
    let sparse_statement =
        single_statement(sparse_commitment, &point_13, dense_13.evaluate(&point_13));
    let mut transcript = Blake2bTranscript::new(b"akita-bb-dense-for-sparse");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &sparse_setup,
            sparse_statement,
            batch_polynomials([&dense_13]),
            sparse_hint,
            &mut transcript,
        ),
        "sparse_unit prover hint requires one-hot witness polynomials",
    );
}
