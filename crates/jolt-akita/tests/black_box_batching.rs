#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{AkitaBlackBoxBatching, AkitaProverHint, AkitaScheme};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, VerifierOpeningClaim,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{
    batch_witness, black_box_setup, black_box_statement, f, layout, polynomial, run_on_large_stack,
    setup_for, single_statement,
};

#[test]
fn akita_black_box_batching_roundtrips_grouped_commitment() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = black_box_setup();
        let poly_a = polynomial(4, 1);
        let poly_b = polynomial(4, 20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = black_box_statement(commitment, &point, [eval_a, eval_b]);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-bb-roundtrip");
        let proof = <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            batch_witness([&poly_a, &poly_b], hint),
            &mut prover_transcript,
        )
        .expect("black-box proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-bb-roundtrip");
        <AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect("black-box proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_black_box_batching_rejects_malformed_statements() {
    run_on_large_stack(|| {
        let (prover_setup, _) = black_box_setup();
        let poly_a = polynomial(4, 1);
        let poly_b = polynomial(4, 20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (group_commitment, group_hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let (other_commitment, _) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[polynomial(4, 80)])
                .expect("other commit should succeed");

        let mut transcript = Blake2bTranscript::new(b"akita-bb-empty");
        assert!(matches!(
            <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                Vec::new(),
                (Vec::new(), AkitaProverHint::default()),
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
            <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                mixed_commitments,
                batch_witness([&poly_a, &poly_b], group_hint.clone()),
                &mut transcript,
            ),
            Err(OpeningsError::InvalidBatch(_))
        ));

        let mut mixed_points =
            black_box_statement(group_commitment.clone(), &point, [eval_a, eval_b]);
        mixed_points[1].evaluation = EvaluationClaim::new(vec![f(11), f(3), f(5), f(7)], eval_b);
        let mut transcript = Blake2bTranscript::new(b"akita-bb-mixed-points");
        assert!(matches!(
            <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                mixed_points,
                batch_witness([&poly_a, &poly_b], group_hint.clone()),
                &mut transcript,
            ),
            Err(OpeningsError::InvalidBatch(_))
        ));

        let one_claim_for_two_slots = single_statement(group_commitment, &point, eval_a);
        let mut transcript = Blake2bTranscript::new(b"akita-bb-claim-count");
        assert!(matches!(
            <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                one_claim_for_two_slots,
                batch_witness([&poly_a, &poly_b], group_hint),
                &mut transcript,
            ),
            Err(OpeningsError::InvalidBatch(_))
        ));
    });
}

#[test]
fn akita_black_box_batching_rejects_bad_prover_witnesses() {
    run_on_large_stack(|| {
        let (prover_setup, _) = black_box_setup();
        let poly_a = polynomial(4, 1);
        let poly_b = polynomial(4, 20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let (_, other_hint) = AkitaScheme::commit_group(
            &prover_setup,
            layout(7),
            &[polynomial(4, 80), polynomial(4, 100)],
        )
        .expect("other grouped commit should succeed");
        let statement = black_box_statement(commitment, &point, [eval_a, eval_b]);

        let mut transcript = Blake2bTranscript::new(b"akita-bb-wrong-hint");
        assert!(
            matches!(
                <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                    &prover_setup,
                    statement.clone(),
                    batch_witness([&poly_a, &poly_b], other_hint),
                    &mut transcript,
                ),
                Err(OpeningsError::InvalidBatch(message)) if message.contains("hint")
            ),
            "mismatched prover hint should reject"
        );

        let mut transcript = Blake2bTranscript::new(b"akita-bb-wrong-count");
        assert!(matches!(
            <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                statement.clone(),
                batch_witness([&poly_a], hint.clone()),
                &mut transcript,
            ),
            Err(OpeningsError::InvalidBatch(_))
        ));

        let wrong_dimension = polynomial(3, 200);
        let mut transcript = Blake2bTranscript::new(b"akita-bb-wrong-dim");
        assert!(matches!(
            <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
                &prover_setup,
                statement,
                batch_witness([&poly_a, &wrong_dimension], hint),
                &mut transcript,
            ),
            Err(OpeningsError::InvalidBatch(_))
        ));
    });
}

#[test]
fn akita_black_box_batching_rejects_tampered_verifier_inputs() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = black_box_setup();
        let poly_a = polynomial(4, 1);
        let poly_b = polynomial(4, 20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = black_box_statement(commitment.clone(), &point, [eval_a, eval_b]);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-bb-tamper");
        let proof = <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            batch_witness([&poly_a, &poly_b], hint),
            &mut prover_transcript,
        )
        .expect("black-box proof should be produced");

        let mut tampered_value = statement.clone();
        tampered_value[0].evaluation.value += f(1);
        assert_black_box_verify_rejects(&verifier_setup, tampered_value, &proof);

        let mut tampered_point = statement.clone();
        tampered_point[1].evaluation = EvaluationClaim::new(vec![f(2), f(11), f(5), f(7)], eval_b);
        assert_black_box_verify_rejects(&verifier_setup, tampered_point, &proof);

        let (other_commitment, _) = AkitaScheme::commit_group(
            &prover_setup,
            layout(7),
            &[polynomial(4, 80), polynomial(4, 100)],
        )
        .expect("other grouped commit should succeed");
        let tampered_commitment = black_box_statement(other_commitment, &point, [eval_a, eval_b]);
        assert_black_box_verify_rejects(&verifier_setup, tampered_commitment, &proof);

        let (_, wrong_layout_setup) = setup_for(4, 2, layout(8));
        assert_black_box_verify_rejects(&wrong_layout_setup, statement, &proof);
    });
}

fn assert_black_box_verify_rejects(
    setup: &<AkitaScheme as CommitmentScheme>::VerifierSetup,
    statement: jolt_akita::AkitaBlackBoxBatchStatement,
    proof: &jolt_akita::AkitaBatchProof,
) {
    let mut transcript = Blake2bTranscript::new(b"akita-bb-tamper");
    assert!(<AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
        setup,
        statement,
        proof,
        &mut transcript,
    )
    .is_err());
}
