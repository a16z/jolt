#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

pub mod support;

use jolt_akita::{AkitaCommitment, AkitaNativeBatching, AkitaProverHint, AkitaScheme};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, VerifierOpeningClaim,
};
use support::{
    assert_transcripts_agree, batch_polynomials, f, layout, native_setup, native_statement,
    new_prover_transcript, new_verifier_transcript, polynomial, setup_for, single_statement,
};

#[test]
fn akita_native_batching_roundtrips_grouped_commitment() {
    let (prover_setup, verifier_setup) = native_setup();
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let statement = native_statement(commitment, &point, [eval_a, eval_b]);

    let mut prover_transcript = new_prover_transcript(b"akita-bb-roundtrip");
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut prover_transcript,
    )
    .expect("black-box proof should be produced");
    let proof = prover_transcript.narg().to_vec();

    let mut verifier_transcript = new_verifier_transcript(b"akita-bb-roundtrip", &proof);
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &statement,
        &mut verifier_transcript,
    )
    .expect("black-box proof should verify");
    assert_transcripts_agree(prover_transcript, verifier_transcript);
}

#[test]
fn akita_native_batching_rejects_malformed_statements() {
    let (prover_setup, _) = native_setup();
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (group_commitment, group_hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let (other_commitment, _) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[polynomial(16, 80)])
            .expect("other commit should succeed");

    let mut transcript = new_prover_transcript(b"akita-bb-empty");
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
    let mut transcript = new_prover_transcript(b"akita-bb-mixed-commit");
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
    let mut transcript = new_prover_transcript(b"akita-bb-mixed-points");
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
    let mut transcript = new_prover_transcript(b"akita-bb-claim-count");
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
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let (_, other_hint) = AkitaScheme::commit_group(
        &prover_setup,
        layout(7),
        &[polynomial(16, 80), polynomial(16, 100)],
    )
    .expect("other grouped commit should succeed");
    let statement = native_statement(commitment, &point, [eval_a, eval_b]);

    let mut transcript = new_prover_transcript(b"akita-bb-wrong-hint");
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

    let mut transcript = new_prover_transcript(b"akita-bb-wrong-count");
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
    let mut transcript = new_prover_transcript(b"akita-bb-wrong-dim");
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
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&point);
    let eval_b = poly_b.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("grouped commit should succeed");
    let statement = native_statement(commitment.clone(), &point, [eval_a, eval_b]);

    let mut prover_transcript = new_prover_transcript(b"akita-bb-tamper");
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut prover_transcript,
    )
    .expect("black-box proof should be produced");
    let proof = prover_transcript.narg().to_vec();

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
        &[polynomial(16, 80), polynomial(16, 100)],
    )
    .expect("other grouped commit should succeed");
    let tampered_commitment = native_statement(other_commitment, &point, [eval_a, eval_b]);
    assert_native_verify_rejects(&verifier_setup, tampered_commitment, &proof);
}

fn assert_native_verify_rejects(
    setup: &<AkitaScheme as CommitmentScheme>::VerifierSetup,
    statement: jolt_akita::AkitaNativeBatchStatement,
    proof: &[u8],
) {
    let mut transcript = new_verifier_transcript(b"akita-bb-tamper", proof);
    assert!(<AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        setup,
        &statement,
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
    let poly = polynomial(16, 1);
    let short_point: Vec<_> = (0..15).map(|index| f(index + 2)).collect();
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
            .expect("commit should succeed");
    let statement = single_statement(commitment, &short_point, f(9));

    let mut transcript = new_prover_transcript(b"akita-bb-short-point");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement,
            batch_polynomials([&poly]),
            hint,
            &mut transcript,
        ),
        "15 variables but commitment has 16",
    );
}

/// The exact-dimension and group-width checks run against the verifier's own
/// setup, so a statement built for one setup must reject against another.
#[test]
fn akita_native_batching_rejects_statements_outside_the_verifier_setup() {
    let (small_setup, _) = setup_for(14, 1, layout(7));
    let small_poly = polynomial(14, 1);
    let small_point: Vec<_> = (0..14).map(|index| f(index + 2)).collect();
    let small_eval = small_poly.evaluate(&small_point);
    let (small_commitment, small_hint) =
        AkitaScheme::commit_group(&small_setup, layout(7), std::slice::from_ref(&small_poly))
            .expect("commit should succeed");
    let small_statement = single_statement(small_commitment, &small_point, small_eval);
    let mut transcript = new_prover_transcript(b"akita-bb-cross-setup");
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &small_setup,
        small_statement.clone(),
        batch_polynomials([&small_poly]),
        small_hint,
        &mut transcript,
    )
    .expect("proof should be produced");
    let small_proof = transcript.narg().to_vec();

    // A 14-variable commitment against a 15-variable verifier setup.
    let (_, wider_verifier) = setup_for(15, 2, layout(7));
    let mut transcript = new_verifier_transcript(b"akita-bb-cross-setup", &small_proof);
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &wider_verifier,
            &small_statement,
            &mut transcript,
        ),
        "does not match exact setup dimension",
    );

    // A two-polynomial group against a verifier setup capped at one slot.
    let (two_slot_setup, _) = native_setup();
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|index| f(index + 2)).collect();
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
    let mut transcript = new_prover_transcript(b"akita-bb-cross-setup");
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &two_slot_setup,
        group_statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        group_hint,
        &mut transcript,
    )
    .expect("group proof should be produced");
    let group_proof = transcript.narg().to_vec();
    let (_, one_slot_verifier) = setup_for(16, 1, layout(7));
    let mut transcript = new_verifier_transcript(b"akita-bb-cross-setup", &group_proof);
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &one_slot_verifier,
            &group_statement,
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
    let poly = polynomial(16, 1);
    let point: Vec<_> = (0..16).map(|index| f(index + 2)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
            .expect("commit should succeed");
    let statement = single_statement(commitment.clone(), &point, eval);
    let mut transcript = new_prover_transcript(b"akita-bb-full-chunk");
    <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly]),
        hint,
        &mut transcript,
    )
    .expect("proof should be produced");
    let proof = transcript.narg().to_vec();

    let mut forged = serde_json::to_value(&commitment).expect("commitment serializes");
    *forged
        .get_mut("one_hot_k")
        .expect("commitment exposes one_hot_k") = serde_json::json!(4);
    let forged: AkitaCommitment =
        serde_json::from_value(forged).expect("forged commitment deserializes");
    let forged_statement = single_statement(forged, &point, eval);

    let mut transcript = new_verifier_transcript(b"akita-bb-full-chunk", &proof);
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &forged_statement,
            &mut transcript,
        ),
        "invalid one-hot metadata",
    );
}

/// One-hot hints certify that the committed data was one-hot; handing the
/// prover dense witnesses for such a hint must reject.
#[test]
fn akita_native_batching_rejects_dense_witnesses_for_one_hot_hints() {
    use jolt_akita::{AkitaScheduleArtifacts, AkitaSetupParams, AKITA_ONE_HOT_K16};
    use jolt_poly::OneHotPolynomial;

    let (one_hot_setup, _) = AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        12,
        1,
        layout(7),
        AKITA_ONE_HOT_K16,
        AkitaScheduleArtifacts::shared_from_default_directory(),
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
    let mut transcript = new_prover_transcript(b"akita-bb-dense-for-onehot");
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
}
