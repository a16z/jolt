#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

use jolt_akita::{AkitaCommitInput, AkitaCommitment, AkitaField, AkitaScheme, AkitaSetupParams};
use jolt_field::Field;
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme, OpeningsError,
    PackedCombine, PhysicalView, ZkBatchOpeningScheme,
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

type PackedAkita = PackedCombine<AkitaScheme>;

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
        layout_digest: layout(7),
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

fn unit_packed_statement(
    commitment: AkitaCommitment,
    point: &[AkitaField],
    eval_a: AkitaField,
    eval_b: AkitaField,
) -> BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: layout(7),
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval_a,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout(7),
                    coefficients: vec![f(1)],
                },
                scale: f(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval_b,
                view: PhysicalView::PackedLinear {
                    layout_digest: layout(7),
                    coefficients: vec![f(1)],
                },
                scale: f(5),
            },
        ],
    }
}

fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(test)
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked");
}

#[test]
fn akita_field_satisfies_jolt_field_bundle() {
    fn assert_field<F: Field>() {}
    assert_field::<AkitaField>();
}

#[test]
fn akita_single_opening_roundtrip() {
    run_on_large_stack(|| {
        let (prover_setup, verifier_setup) = setup();
        let poly = polynomial(1);
        let point = vec![f(2), f(3), f(5), f(7)];
        let eval = poly.evaluate(&point);
        let (commitment, hint) = AkitaScheme::commit_packed_witness(
            &prover_setup,
            AkitaCommitInput {
                layout_digest: layout(7),
                polynomial: poly.clone(),
            },
        )
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
fn packed_combine_akita_unit_packed_views_roundtrip() {
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
        let statement = unit_packed_statement(commitment.clone(), &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-combine");
        let proof = <PackedAkita as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("packed wrapper should produce an Akita proof");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-combine");
        let result = <PackedAkita as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed wrapper should verify through Akita");

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
fn packed_combine_akita_binds_packed_coefficients_to_native_proof() {
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
        let statement = unit_packed_statement(commitment, &point, eval_a, eval_b);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-coeff-tamper");
        let proof = <PackedAkita as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            &mut prover_transcript,
            &statement,
            &[poly_a, poly_b],
            vec![hint],
        )
        .expect("packed proof should be produced");

        let mut tampered = statement;
        tampered.claims[0].view = PhysicalView::PackedLinear {
            layout_digest: layout(7),
            coefficients: vec![f(0), f(1)],
        };
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-coeff-tamper");
        let result = <PackedAkita as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &tampered,
            &proof,
        );
        assert!(result.is_err(), "changed packed coefficients should reject");
    });
}

#[test]
fn akita_native_adapter_rejects_packed_linear_view_until_lowered() {
    let (prover_setup, _) = setup();
    let poly = polynomial(1);
    let point = vec![f(2), f(3), f(5), f(7)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit_packed_witness(
        &prover_setup,
        AkitaCommitInput {
            layout_digest: layout(7),
            polynomial: poly.clone(),
        },
    )
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
            view: PhysicalView::PackedLinear {
                layout_digest: layout(7),
                coefficients: vec![f(1)],
            },
            scale: f(1),
        }],
    };

    let mut transcript = Blake2bTranscript::new(b"akita-packed");
    let result = <AkitaScheme as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut transcript,
        &statement,
        &[poly],
        vec![hint],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
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
