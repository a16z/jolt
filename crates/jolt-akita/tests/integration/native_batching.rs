#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

use std::sync::Arc;

use crate::support::{
    batch_polynomials, f, layout, native_setup, native_statement, polynomial, setup_for,
    single_statement,
};
use akita_params::PolynomialGroupLayout;
use jolt_akita::{
    AkitaBatchProof, AkitaChunkProfile, AkitaCommitment, AkitaNativeBatching, AkitaProverHint,
    AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
    GroupedScheduleParams, TraceOneHotRows, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, GroupBatchOpening, GroupOpeningClaim,
    OpeningsError, VerifierOpeningClaim,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::{protocol_tag, prover_transcript, verifier_transcript, Blake2b512, Keccak};

struct TraceRows {
    indices: Vec<[u8; 2]>,
}

impl TraceOneHotRows for TraceRows {
    fn num_rows(&self) -> usize {
        self.indices.len()
    }

    fn num_columns(&self) -> usize {
        2
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        selected_rows.copy_from_slice(&self.indices[row]);
    }

    fn committed_digit_zero_mask(&self, _row: usize) -> u64 {
        2
    }
}

#[test]
fn akita_streamed_trace_batch_rejects_swapped_evaluations() {
    const NUM_VARS: usize = 16;
    let (prover_setup, verifier_setup) =
        AkitaScheme::setup(AkitaSetupParams::one_hot_only_grouped(
            NUM_VARS,
            2,
            2,
            layout(7),
            AKITA_ONE_HOT_K16,
            Some(GroupedScheduleParams::new(
                None,
                None,
                Vec::new(),
                PolynomialGroupLayout::new(NUM_VARS, 2),
            )),
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        ))
        .expect("streamed trace setup should build");
    let indices: Vec<_> = (0..1 << (NUM_VARS - 4))
        .map(|row| [(row % 16) as u8, ((row * 3 + 1) % 16) as u8])
        .collect();
    let point: Vec<_> = (0..NUM_VARS).map(|index| f(2 + 3 * index as u64)).collect();
    let evaluations = (0..2)
        .map(|column| {
            let polynomial = OneHotPolynomial::new(
                AKITA_ONE_HOT_K16,
                indices
                    .iter()
                    .map(|row| (row[column] != 0 || column == 1).then_some(row[column]))
                    .collect(),
            );
            polynomial.evaluate(&point)
        })
        .collect::<Vec<_>>();
    assert_ne!(evaluations[0], evaluations[1]);
    let (commitment, hint) = AkitaScheme::commit_trace_one_hot(
        &prover_setup,
        layout(7),
        Arc::new(TraceRows { indices }),
        &[],
    )
    .expect("streamed trace group should commit");
    let mut statement = GroupOpeningClaim::new(commitment, point, evaluations);
    let session = protocol_tag(b"akita-streamed-tamper");
    let mut prover_state = prover_transcript::<Keccak>(&session, b"");
    let proof = AkitaScheme::prove_batch(
        &prover_setup,
        Vec::new(),
        statement.clone(),
        hint,
        &mut prover_state,
    )
    .expect("streamed trace batch should prove");
    let narg = prover_state.narg_string().to_vec();
    let prover_tail = prover_state.verifier_message::<[u8; 32]>();
    let verify_proof = |setup: &AkitaVerifierSetup,
                        statement: &GroupOpeningClaim<_, _>,
                        proof: &AkitaBatchProof,
                        narg: &[u8]| {
        let mut verifier_state = verifier_transcript::<Keccak>(&session, b"", narg);
        let pending = AkitaScheme::verify_batch(setup, &[], statement, proof, &mut verifier_state)?;
        let tail = verifier_state.verifier_message::<[u8; 32]>();
        AkitaScheme::check_eof(pending, verifier_state).map(|()| tail)
    };
    let verify_under =
        |setup: &AkitaVerifierSetup, statement: &GroupOpeningClaim<_, _>, narg: &[u8]| {
            verify_proof(setup, statement, &proof, narg)
        };
    let verify = |statement: &GroupOpeningClaim<_, _>, narg: &[u8]| {
        verify_under(&verifier_setup, statement, narg)
    };
    assert_eq!(
        verify(&statement, &narg).expect("original streamed trace evaluations must verify"),
        prover_tail,
        "prover and verifier sponges must agree after the opening"
    );

    // The trace batch's argument lives in Jolt's string, so a proof that also
    // carries a backend proof must reject.
    let mut carrying = serde_json::to_value(&proof).expect("proof should serialize");
    *carrying
        .get_mut("backend_proof")
        .expect("proof should expose backend_proof") = serde_json::json!([0]);
    let carrying: AkitaBatchProof =
        serde_json::from_value(carrying).expect("changed proof should deserialize");
    let err = verify_proof(&verifier_setup, &statement, &carrying, &narg)
        .expect_err("a non-empty backend proof must reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("standalone argument")),
        "expected the embedded-argument rejection, got: {err}"
    );

    // The embedded opening absorbs no setup bytes of its own. Akita's
    // descriptor binds the backend setup and the selected schedule, and the
    // trace batch must match the verifier setup's shape, layout digest and K
    // exactly, so a setup that differs in any of them rejects the proof.
    let setup_json =
        serde_json::to_value(&verifier_setup).expect("verifier setup should serialize");
    for (field, value) in [
        ("max_num_vars", serde_json::json!(NUM_VARS + 1)),
        ("max_num_polys_per_commitment_group", serde_json::json!(3)),
        ("default_layout_digest", serde_json::json!(layout(8))),
        ("one_hot_k", serde_json::json!(AKITA_ONE_HOT_K256)),
    ] {
        let mut changed = setup_json.clone();
        *changed.get_mut(field).expect("verifier setup field") = value;
        let changed: AkitaVerifierSetup =
            serde_json::from_value(changed).expect("changed verifier setup should deserialize");
        assert!(
            verify_under(&changed, &statement, &narg).is_err(),
            "a verifier setup with a different {field} must reject"
        );
    }

    // The argument lives in the caller's string, so its end is the string's
    // end: one appended byte must reject.
    let mut extended = narg.clone();
    extended.push(0);
    assert!(verify(&statement, &extended).is_err());

    // A forged coefficient count must reject at the trusted shape check,
    // before the decoder can reserve a payload-sized buffer.
    let mut forged =
        serde_json::to_value(&statement.commitment).expect("commitment should serialize");
    *forged
        .get_mut("backend_coeff_len")
        .expect("commitment should expose backend_coeff_len") = (1u64 << 25).into();
    let mut forged_statement = statement.clone();
    forged_statement.commitment =
        serde_json::from_value(forged).expect("forged commitment should deserialize");
    let err = verify(&forged_statement, &narg).expect_err("forged backend_coeff_len should reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("coefficients")),
        "expected a shape-guard rejection, got: {err}"
    );

    statement.evaluations.swap(0, 1);
    assert!(verify(&statement, &narg).is_err());
}

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

    let mut prover_state =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-roundtrip"), b"");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut prover_state,
    )
    .expect("black-box proof should be produced");

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-roundtrip"), b"");
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("black-box proof should verify");
    assert_eq!(
        prover_state.verifier_message::<[u8; 32]>(),
        verifier_transcript.verifier_message::<[u8; 32]>()
    );
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

    let mut transcript = prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-empty"), b"");
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-mixed-commit"), b"");
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-mixed-points"), b"");
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-claim-count"), b"");
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

    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-wrong-hint"), b"");
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

    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-wrong-count"), b"");
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
    let mut transcript = prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-wrong-dim"), b"");
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

    let mut prover_state = prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-tamper"), b"");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        hint,
        &mut prover_state,
    )
    .expect("black-box proof should be produced");

    let mut verifier_transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-tamper"), b"");
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("original ordered evaluations must verify");
    assert_ne!(eval_a, eval_b);
    let swapped_values = native_statement(commitment.clone(), &point, [eval_b, eval_a]);
    assert_native_verify_rejects(&verifier_setup, swapped_values, &proof);

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

fn assert_native_verify_rejects(
    setup: &<AkitaScheme as CommitmentScheme>::VerifierSetup,
    statement: jolt_akita::AkitaNativeBatchStatement,
    proof: &jolt_akita::AkitaBatchProof,
) {
    let mut transcript = prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-tamper"), b"");
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
    let poly = polynomial(16, 1);
    let short_point: Vec<_> = (0..15).map(|index| f(index + 2)).collect();
    let (commitment, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
            .expect("commit should succeed");
    let statement = single_statement(commitment, &short_point, f(9));

    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-short-point"), b"");
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-cross-setup"), b"");
    let small_proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &small_setup,
        small_statement.clone(),
        batch_polynomials([&small_poly]),
        small_hint,
        &mut transcript,
    )
    .expect("proof should be produced");

    let (_, wider_verifier) = setup_for(15, 2, layout(7));
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-cross-setup"), b"");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &wider_verifier,
            &small_statement,
            &small_proof,
            &mut transcript,
        ),
        "does not match exact setup dimension",
    );

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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-cross-setup"), b"");
    let group_proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &two_slot_setup,
        group_statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        group_hint,
        &mut transcript,
    )
    .expect("group proof should be produced");
    let (_, one_slot_verifier) = setup_for(16, 1, layout(7));
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-cross-setup"), b"");
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-full-chunk"), b"");
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
    let forged: AkitaCommitment =
        serde_json::from_value(forged).expect("forged commitment deserializes");
    let forged_statement = single_statement(forged, &point, eval);

    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-full-chunk"), b"");
    expect_invalid_batch(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &forged_statement,
            &proof,
            &mut transcript,
        ),
        "invalid one-hot metadata",
    );
}

/// One-hot hints certify that the committed data was one-hot; handing the
/// prover dense witnesses for such a hint must reject.
#[test]
fn akita_native_batching_rejects_dense_witnesses_for_one_hot_hints() {
    let (one_hot_setup, _) = AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        12,
        1,
        layout(7),
        AKITA_ONE_HOT_K16,
        AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
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
    let mut transcript =
        prover_transcript::<Blake2b512>(&protocol_tag(b"akita-bb-dense-for-onehot"), b"");
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
