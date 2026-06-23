#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{AkitaScheme, AkitaSetupParams};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PhysicalView,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{
    direct_statement, f, layout, polynomial, run_on_large_stack, setup, OpeningId, RelationId,
};

#[test]
fn akita_setup_key_is_bound_to_batch_proof() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let (_, wrong_layout_setup) = AkitaScheme::setup(AkitaSetupParams::new(4, 2, layout(8)));
        let (_, wrong_dimension_setup) = AkitaScheme::setup(AkitaSetupParams::new(5, 2, layout(7)));
        let (_, wrong_group_setup) = AkitaScheme::setup(AkitaSetupParams::new(4, 3, layout(7)));
        let poly = polynomial(1);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval = poly.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
                .expect("commit should succeed");
        let statement = BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point,
            layout_digest: layout(7),
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::NativeBatch,
                commitment,
                claim: eval,
                view: PhysicalView::Direct,
                scale: f(1),
            }],
        };

        let mut prover_transcript = Blake2bTranscript::new(b"akita-setup-key");
        let proof = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            std::slice::from_ref(&poly),
            vec![hint],
        )
        .expect("proof should be produced");

        for setup in [
            &wrong_layout_setup,
            &wrong_dimension_setup,
            &wrong_group_setup,
        ] {
            let mut verifier_transcript = Blake2bTranscript::new(b"akita-setup-key");
            let result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
                setup,
                &mut verifier_transcript,
                &statement,
                &proof,
            );
            assert!(result.is_err(), "wrong setup key should reject");
        }

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-setup-key");
        let _result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("matching setup key should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_prover_hint_is_bound_to_statement_commitment() {
    run_on_large_stack(|| {
        let (prover_setup, _) = setup();
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let other_a = polynomial(40);
        let other_b = polynomial(60);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, _) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let (_, other_hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[other_a, other_b])
                .expect("other grouped commit should succeed");
        let statement = direct_statement(commitment, &point, eval_a, eval_b);

        let mut transcript = Blake2bTranscript::new(b"akita-hint-binding");
        let result = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut transcript,
            &statement,
            &[poly_a, poly_b],
            vec![other_hint],
        );

        assert!(
            matches!(result, Err(OpeningsError::InvalidBatch(message)) if message.contains("hint")),
            "mismatched prover hint should reject"
        );
    });
}

#[test]
fn akita_batch_opening_rejects_tampered_claim() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = direct_statement(commitment, &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-tamper");
        let proof = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("batch proof should be produced");

        let mut tampered = statement;
        tampered.claims[0].claim += f(1);
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-tamper");
        let result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "changed claim should reject");
    });
}

#[test]
fn akita_batch_opening_rejects_tampered_commitment() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let other_a = polynomial(40);
        let other_b = polynomial(60);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let (other_commitment, _) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[other_a, other_b])
                .expect("other grouped commit should succeed");
        let statement = direct_statement(commitment, &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-commitment-tamper");
        let proof = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("batch proof should be produced");

        let tampered = direct_statement(other_commitment, &point, eval_a, eval_b);
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-commitment-tamper");
        let result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "changed commitment should reject");
    });
}

#[test]
fn akita_batch_opening_rejects_tampered_scale() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly_a = polynomial(1);
        let poly_b = polynomial(20);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&point);
        let eval_b = poly_b.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
                .expect("grouped commit should succeed");
        let statement = direct_statement(commitment, &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-scale-tamper");
        let proof = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("batch proof should be produced");

        let mut tampered = statement;
        tampered.claims[0].scale += f(1);
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-scale-tamper");
        let result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "changed scale should reject");
    });
}
