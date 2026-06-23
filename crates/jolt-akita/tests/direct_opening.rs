#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheme, AkitaSetupParams};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PhysicalView, ZkBatchOpeningScheme,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationId {
    Packed,
}

fn f(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn polynomial(offset: u64) -> Polynomial<AkitaField> {
    Polynomial::new((0..16).map(|value| f(value + offset)).collect())
}

fn setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::new(4, 2, layout(7)))
}

fn direct_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: commitment.layout_digest,
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::Direct,
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval_b,
                view: PhysicalView::Direct,
                scale: f(5),
            },
        ],
    }
}

fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("test thread should spawn")
        .join()
        .expect("test thread should complete");
}

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
    assert_eq!(statement_layout.0.layout_digest, layout(8));

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
        assert_eq!(commitment.layout_digest, commitment_layout);
        let statement = BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point.clone(),
            layout_digest: commitment_layout,
            claims: vec![BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
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
                relation: RelationId::Packed,
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
                relation: RelationId::Packed,
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

#[test]
fn akita_batch_zk_is_explicitly_unsupported() {
    let (prover_setup, _) = setup();
    let statement = BatchOpeningStatement {
        logical_point: vec![f(1)],
        pcs_point: vec![f(1)],
        layout_digest: layout(7),
        claims: Vec::<BatchOpeningClaim<_, _, OpeningId, RelationId, ()>>::new(),
    };
    let mut transcript = Blake2bTranscript::new(b"akita-zk");
    let result = <AkitaScheme as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        &mut transcript,
        &statement,
        &[],
        &[],
        vec![],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
