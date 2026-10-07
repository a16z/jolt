#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

pub mod support;

use std::sync::Arc;

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_schedules::ValidatedScheduleCatalog;
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, GroupCommitPhaseParams,
    PolynomialGroupLayout,
};
use jolt_akita::configs::{JoltDenseBounded, JoltFieldDigits, JoltSignedBytes};
use jolt_akita::{
    AkitaField, AkitaNativeBatching, AkitaProverHint, AkitaScheduleArtifacts, AkitaScheme,
    AkitaSetupParams, AkitaVerifierSetup, PrecommittedScheduleParams, TraceCommitmentBackend,
};
use jolt_field::Ring;
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, GroupOpeningClaim, OpeningsError,
    PrecommittedClaim, PrecommittedRole, TransparentObjectSetup, VerifierOpeningClaim,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use support::{
    batch_polynomials, f, layout, native_setup, native_statement, polynomial, setup_for,
    single_statement,
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
    let poly_a = polynomial(16, 1);
    let poly_b = polynomial(16, 20);
    let point: Vec<_> = (0..16).map(|i| f(2 + 3 * i)).collect();
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
        &[polynomial(16, 80), polynomial(16, 100)],
    )
    .expect("other grouped commit should succeed");
    let tampered_commitment = native_statement(other_commitment, &point, [eval_a, eval_b]);
    assert_native_verify_rejects(&verifier_setup, tampered_commitment, &proof);

    let (_, wrong_layout_setup) = setup_for(16, 2, layout(8));
    assert_native_verify_rejects(&wrong_layout_setup, statement, &proof);
}

fn artifacts_with_field_digit_groups(
    groups: &[PolynomialGroupLayout],
) -> Arc<AkitaScheduleArtifacts> {
    let rows = groups
        .par_iter()
        .map(|&group| {
            let schedule = find_schedule(
                &AkitaScheduleLookupKey::single(group),
                honest_fold_policy_of::<JoltFieldDigits>(),
                &[],
                &policy_of::<JoltFieldDigits>(),
                JoltFieldDigits::ring_challenge_config,
            )
            .expect("field-digit schedule")
            .schedule;
            let profile = GroupCommitPhaseParams::try_from_params(group, &schedule.root.params)
                .expect("field-digit profile");
            let profiles = CommittedGroupBatchProfile {
                final_group: profile,
                precommitteds: Vec::new(),
            };
            (profiles, schedule)
        })
        .collect::<Vec<_>>();
    let field_digits = ValidatedScheduleCatalog::try_new(
        JoltFieldDigits::schedule_family_name(),
        rows,
        &policy_of::<JoltFieldDigits>(),
        JoltFieldDigits::ring_challenge_config,
    )
    .expect("field-digit catalog")
    .to_artifact_bytes()
    .expect("encode the field-digit catalog");
    let packaged = |family: &str| {
        std::fs::read(AkitaScheduleArtifacts::packaged_directory().join(format!("{family}.aks")))
            .expect("packaged catalog")
    };
    Arc::new(AkitaScheduleArtifacts::new(
        packaged(JoltDenseBounded::schedule_family_name()),
        Vec::new(),
        Vec::new(),
        packaged(JoltSignedBytes::schedule_family_name()),
        field_digits,
    ))
}

#[test]
fn signed_byte_trace_opens_beside_advice_and_late_field_digit_groups() {
    const TRACE_NUM_VARS: usize = 17;
    const ADVICE_NUM_VARS: usize = 18;
    let field_digit_groups = [(14, 2), (12, 1)];
    let artifacts = artifacts_with_field_digit_groups(
        &field_digit_groups
            .map(|(num_vars, num_polys)| PolynomialGroupLayout::new(num_vars, num_polys)),
    );
    let (prover_setup, verifier_setup) =
        AkitaScheme::setup(AkitaSetupParams::signed_bytes_grouped(
            PrecommittedScheduleParams::new(None, Some(ADVICE_NUM_VARS), TRACE_NUM_VARS)
                .with_field_digit_groups(field_digit_groups),
            layout(9),
            Arc::clone(&artifacts),
        ))
        .expect("signed-byte grouped setup");

    let advice = polynomial(ADVICE_NUM_VARS, 7);
    let (advice_setup, _) =
        AkitaScheme::transparent_object_setup(&artifacts, ADVICE_NUM_VARS, layout(8))
            .expect("advice object setup");
    let (advice_commitment, advice_hint) =
        AkitaScheme::commit(&advice, &advice_setup).expect("commit the advice");

    let bytes = (0..1usize << TRACE_NUM_VARS)
        .map(|index| (index.wrapping_mul(37) % 251) as u8 as i8)
        .collect::<Vec<_>>();
    let trace = Polynomial::new(bytes.iter().copied().map(AkitaField::from_i8).collect());
    let (trace_commitment, trace_hint) = AkitaScheme::commit_signed_byte_trace(
        &TraceCommitmentBackend::cpu(),
        &prover_setup,
        layout(9),
        bytes,
        &[&advice_hint],
    )
    .expect("the trace commits before any field-digit group exists");

    let advice_point = (0..ADVICE_NUM_VARS as u64)
        .map(|i| f(3 + 5 * i))
        .collect::<Vec<_>>();
    let advice_claim = GroupOpeningClaim::new(
        advice_commitment,
        advice_point.clone(),
        vec![advice.evaluate(&advice_point)],
    );
    let advice_role = PrecommittedRole::new(0, b"trusted_advice", "trusted-advice");
    let mut openings = vec![(
        PrecommittedClaim::new(advice_role, advice_claim),
        advice_hint,
    )];
    for (order, (num_vars, num_polys)) in (1u64..).zip(field_digit_groups) {
        let tables = (0..num_polys as u64)
            .map(|poly| polynomial(num_vars, 1000 * order + 100 * poly))
            .collect::<Vec<_>>();
        let values = tables
            .iter()
            .map(|table| table.evaluations().to_vec())
            .collect::<Vec<_>>();
        let (commitment, hint) =
            AkitaScheme::commit_field_digit_group(&prover_setup, layout(10), &values)
                .expect("commit a field-digit group");
        let point = (0..num_vars as u64)
            .map(|i| f(5 + 11 * order + 3 * i))
            .collect::<Vec<_>>();
        let evaluations = tables.iter().map(|table| table.evaluate(&point)).collect();
        let role = PrecommittedRole::new(order, b"field_digits", "field-digits");
        let claim = GroupOpeningClaim::new(commitment, point, evaluations);
        openings.push((PrecommittedClaim::new(role, claim), hint));
    }
    let claims = openings
        .iter()
        .map(|(claim, _)| claim.clone())
        .collect::<Vec<_>>();
    let trace_point = (0..TRACE_NUM_VARS as u64)
        .map(|i| f(7 + 2 * i))
        .collect::<Vec<_>>();
    let trace_claim = GroupOpeningClaim::new(
        trace_commitment,
        trace_point.clone(),
        vec![trace.evaluate(&trace_point)],
    );

    let mut prover_transcript = Blake2bTranscript::new(b"akita-signed-byte-grouped");
    let proof = <AkitaScheme as CommitmentScheme>::prove_batch(
        &prover_setup,
        openings,
        trace_claim.clone(),
        trace_hint,
        &mut prover_transcript,
    )
    .expect("grouped signed-byte proof");
    // The verifier re-derives its signed-byte backend key from the
    // transported shape instead of the cache primed at setup.
    let (verifier_setup, _): (AkitaVerifierSetup, usize) = bincode::serde::decode_from_slice(
        &bincode::serde::encode_to_vec(&verifier_setup, bincode::config::standard())
            .expect("encode the verifier setup"),
        bincode::config::standard(),
    )
    .expect("decode the verifier setup");
    let mut verifier_transcript = Blake2bTranscript::new(b"akita-signed-byte-grouped");
    <AkitaScheme as CommitmentScheme>::verify_batch(
        &verifier_setup,
        &claims,
        &trace_claim,
        &proof,
        &mut verifier_transcript,
    )
    .expect("grouped signed-byte proof verifies");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
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
