#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{AkitaScheme, AkitaSetupParams};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PhysicalView,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{
    direct_statement, f, layout, polynomial, run_on_large_stack, setup, OpeningId, RelationId,
};

#[test]
fn akita_single_opening_roundtrip() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly = polynomial(1);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval = poly.evaluate(&point);
        let (commitment, hint) =
            AkitaScheme::commit_group(&prover_setup, layout(7), std::slice::from_ref(&poly))
                .expect("commit should succeed");

        let mut prover_transcript = Blake2bTranscript::new(b"akita-single");
        let proof = AkitaScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-single");
        AkitaScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .expect("single proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_batch_opening_roundtrip_direct_grouped_commitment() {
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
        let statement = direct_statement(commitment.clone(), &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-direct");
        let proof = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("batch proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-direct");
        let result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("batch proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients, vec![f(2), f(5)]);
        assert_eq!(result.reduced_opening, f(2) * eval_a + f(5) * eval_b);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_commit_group_rejects_invalid_shapes() {
    let (prover_setup, _) = setup();
    let poly_a = polynomial(1);
    let mixed_vars = Polynomial::new((0..8).map(|value| f(value + 40)).collect());

    let mixed_result =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), mixed_vars]);
    assert!(matches!(mixed_result, Err(OpeningsError::InvalidBatch(_))));

    let too_many_result = AkitaScheme::commit_group(
        &prover_setup,
        layout(7),
        &[poly_a.clone(), polynomial(20), polynomial(40)],
    );
    assert!(matches!(
        too_many_result,
        Err(OpeningsError::InvalidBatch(_))
    ));
}

#[test]
fn akita_direct_commit_group_accepts_statement_layout_and_rejects_dimension_mismatch() {
    let (prover_setup, _) = setup();
    let statement_layout = AkitaScheme::commit_group(&prover_setup, layout(8), &[polynomial(1)])
        .expect("direct commitments carry their statement layout digest");
    assert_eq!(statement_layout.0.layout_digest(), layout(8));

    let (wrong_dimension_setup, _) = AkitaScheme::setup(AkitaSetupParams::new(5, 2, layout(7)));
    let wrong_dimension =
        AkitaScheme::commit_group(&wrong_dimension_setup, layout(7), &[polynomial(1)]);
    assert!(matches!(
        wrong_dimension,
        Err(OpeningsError::InvalidBatch(_))
    ));
}

#[test]
fn akita_direct_opening_uses_commitment_layout_digest() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly = polynomial(1);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval = poly.evaluate(&point);
        let commitment_layout = layout(11);
        let unrelated_statement_layout = layout(7);
        let (commitment, hint) = AkitaScheme::commit_group(
            &prover_setup,
            commitment_layout,
            std::slice::from_ref(&poly),
        )
        .expect("direct commitment should commit with its own layout digest");
        assert_eq!(commitment.layout_digest(), commitment_layout);
        let statement = BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point.clone(),
            layout_digest: commitment_layout,
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::NativeBatch,
                commitment: commitment.clone(),
                claim: eval,
                view: PhysicalView::Direct,
                scale: f(1),
            }],
        };

        let mut prover_transcript = Blake2bTranscript::new(b"akita-direct-commitment-layout");
        let proof = <AkitaScheme as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            std::slice::from_ref(&poly),
            vec![hint],
        )
        .expect("direct proof should use commitment layout digest");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-direct-commitment-layout");
        let result = <AkitaScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("direct proof should verify with commitment layout digest");
        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mismatched_statement = BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point,
            layout_digest: unrelated_statement_layout,
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::NativeBatch,
                commitment,
                claim: eval,
                view: PhysicalView::Direct,
                scale: f(1),
            }],
        };
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-direct-commitment-layout");
        assert!(
            matches!(
                <AkitaScheme as BatchOpeningScheme>::verify_batch(
                    &verifier_setup,
                    &mut verifier_transcript,
                    &mismatched_statement,
                    &proof,
                ),
                Err(OpeningsError::InvalidBatch(reason))
                    if reason.contains("layout digest")
            ),
            "direct proof must reject a statement digest that differs from the opened commitment"
        );
    });
}
