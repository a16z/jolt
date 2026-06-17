#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

use jolt_akita::{
    to_akita_claim, AkitaCommitInput, AkitaCommitment, AkitaScheme, AkitaSetupParams,
};
use jolt_field::{CanonicalBytes, FixedByteSize, Fr, FromPrimitiveInt, Invertible};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, OpeningsError, PhysicalView,
    ZkBatchOpeningScheme,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

type AkitaFr = AkitaScheme<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationId {
    Packed,
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn layout(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn polynomial() -> Polynomial<Fr> {
    Polynomial::new((0..8).map(|value| fr(value + 1)).collect())
}

fn setup() -> (
    <AkitaFr as jolt_openings::CommitmentScheme>::ProverSetup,
    <AkitaFr as jolt_openings::CommitmentScheme>::VerifierSetup,
) {
    <AkitaFr as jolt_openings::CommitmentScheme>::setup(AkitaSetupParams::exact(3, layout(7)))
}

fn commit(
    prover_setup: &<AkitaFr as jolt_openings::CommitmentScheme>::ProverSetup,
    poly: &Polynomial<Fr>,
) -> (
    AkitaCommitment,
    <AkitaFr as jolt_openings::CommitmentScheme>::OpeningHint,
) {
    AkitaFr::commit_packed_witness(
        prover_setup,
        AkitaCommitInput {
            layout_digest: layout(7),
            d_pack: 3,
            evaluations: poly.evals().to_vec(),
        },
    )
    .expect("commit should accept matching dimension")
}

fn direct_statement(
    commitment: AkitaCommitment,
    point: &[Fr],
    eval: Fr,
) -> BatchOpeningStatement<Fr, AkitaCommitment, OpeningId, RelationId> {
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: layout(7),
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::Packed,
                commitment: commitment.clone(),
                claim: eval,
                view: PhysicalView::Direct,
                scale: fr(2),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Packed,
                commitment,
                claim: eval,
                view: PhysicalView::Direct,
                scale: fr(5),
            },
        ],
    }
}

#[test]
fn akita_batch_opening_roundtrip_direct_non_homomorphic() {
    let (prover_setup, verifier_setup) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let statement = direct_statement(commitment.clone(), &point, eval);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-direct");
    let proof = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut prover_transcript,
        &statement,
        &[poly],
        vec![hint],
    )
    .expect("batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-direct");
    let result = <AkitaFr as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("batch proof should verify");

    assert_eq!(result.joint_commitment, commitment);
    assert_eq!(result.coefficients, vec![fr(2), fr(5)]);
    assert_eq!(result.reduced_opening, fr(7) * eval);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn akita_batch_opening_roundtrip_packed_linear_view() {
    let (prover_setup, verifier_setup) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let decode = fr(7);
    let logical_claim = eval * decode.inverse().expect("decode is nonzero");
    let (commitment, hint) = commit(&prover_setup, &poly);
    let statement = BatchOpeningStatement {
        logical_point: point.clone(),
        pcs_point: point,
        layout_digest: layout(7),
        claims: vec![BatchOpeningClaim {
            id: OpeningId::A,
            relation: RelationId::Packed,
            commitment: commitment.clone(),
            claim: logical_claim,
            view: PhysicalView::PackedLinear {
                layout_digest: layout(7),
                coefficients: vec![decode],
            },
            scale: fr(3),
        }],
    };

    let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-linear");
    let proof = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut prover_transcript,
        &statement,
        &[poly],
        vec![hint],
    )
    .expect("packed-linear proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-linear");
    let result = <AkitaFr as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("packed-linear proof should verify");

    assert_eq!(result.joint_commitment, commitment);
    assert_eq!(result.coefficients, vec![fr(21)]);
    assert_eq!(result.reduced_opening, fr(21) * logical_claim);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn akita_layout_digest_is_bound() {
    let (prover_setup, verifier_setup) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let statement = direct_statement(commitment, &point, eval);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-layout");
    let proof = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut prover_transcript,
        &statement,
        &[poly],
        vec![hint],
    )
    .expect("batch proof should be produced");

    let mut tampered = statement;
    tampered.layout_digest = layout(8);

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-layout");
    let result = <AkitaFr as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &tampered,
        &proof,
    );
    assert!(result.is_err(), "changed layout digest should reject");
}

#[test]
fn akita_batch_opening_rejects_tampered_claim() {
    let (prover_setup, verifier_setup) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let statement = direct_statement(commitment, &point, eval);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-claim");
    let proof = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut prover_transcript,
        &statement,
        &[poly],
        vec![hint],
    )
    .expect("batch proof should be produced");

    let mut tampered = statement;
    tampered.claims[0].claim += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-claim");
    let result = <AkitaFr as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &tampered,
        &proof,
    );
    assert!(result.is_err(), "changed claim should reject");
}

#[test]
fn akita_batch_opening_rejects_tampered_commitment() {
    let (prover_setup, verifier_setup) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let statement = direct_statement(commitment, &point, eval);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-commitment");
    let proof = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut prover_transcript,
        &statement,
        &[poly],
        vec![hint],
    )
    .expect("batch proof should be produced");

    let mut tampered = statement;
    tampered.claims[0].commitment.commitment_digest = layout(9);

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-commitment");
    let result = <AkitaFr as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &mut verifier_transcript,
        &tampered,
        &proof,
    );
    assert!(result.is_err(), "changed commitment should reject");
}

#[test]
fn akita_wrong_dimension_rejects() {
    let (prover_setup, _) = setup();
    let result = AkitaFr::commit_packed_witness(
        &prover_setup,
        AkitaCommitInput {
            layout_digest: layout(7),
            d_pack: 2,
            evaluations: polynomial().evals().to_vec(),
        },
    );
    assert!(matches!(result, Err(OpeningsError::InvalidSetup(_))));
}

#[test]
fn akita_statement_has_single_packed_witness_commitment() {
    let (prover_setup, _) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let mut second_commitment = commitment.clone();
    second_commitment.commitment_digest = layout(11);
    let mut statement = direct_statement(commitment, &point, eval);
    statement.claims[1].commitment = second_commitment;

    let mut transcript = Blake2bTranscript::new(b"akita-single-commitment");
    let result = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut transcript,
        &statement,
        &[poly],
        vec![hint],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn akita_hint_layout_mismatch_rejects() {
    let (prover_setup, _) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, mut hint) = commit(&prover_setup, &poly);
    hint.layout_digest = layout(12);
    let statement = direct_statement(commitment, &point, eval);

    let mut transcript = Blake2bTranscript::new(b"akita-hint");
    let result = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut transcript,
        &statement,
        &[poly],
        vec![hint],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn akita_unsupported_view_formula_rejects() {
    let (prover_setup, _) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let mut statement = direct_statement(commitment, &point, eval);
    statement.claims[0].view = PhysicalView::PackedLinear {
        layout_digest: layout(7),
        coefficients: Vec::new(),
    };

    let mut transcript = Blake2bTranscript::new(b"akita-unsupported-view");
    let result = <AkitaFr as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        &mut transcript,
        &statement,
        &[poly],
        vec![hint],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn akita_transparent_rejects_zk_mode() {
    let (prover_setup, verifier_setup) = setup();
    let poly = polynomial();
    let point = vec![fr(2), fr(3), fr(5)];
    let eval = poly.evaluate(&point);
    let (commitment, hint) = commit(&prover_setup, &poly);
    let clear_statement = direct_statement(commitment, &point, eval);
    let zk_statement = BatchOpeningStatement {
        logical_point: clear_statement.logical_point,
        pcs_point: clear_statement.pcs_point,
        layout_digest: clear_statement.layout_digest,
        claims: clear_statement
            .claims
            .into_iter()
            .map(|claim| BatchOpeningClaim {
                id: claim.id,
                relation: claim.relation,
                commitment: claim.commitment,
                claim: (),
                view: claim.view,
                scale: claim.scale,
            })
            .collect(),
    };

    let mut transcript = Blake2bTranscript::new(b"akita-zk");
    let prove_result = <AkitaFr as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        &mut transcript,
        &zk_statement,
        &[eval],
        &[poly],
        vec![hint],
    );
    assert!(matches!(prove_result, Err(OpeningsError::InvalidBatch(_))));

    let mut transcript = Blake2bTranscript::new(b"akita-zk");
    let dummy_proof = jolt_akita::AkitaBatchProof {
        setup_key: verifier_setup.key.clone(),
        packed_commitment: zk_statement.claims[0].commitment.clone(),
        statement_digest: fr(0),
        coefficients: Vec::new(),
        reduced_opening: fr(0),
    };
    let verify_result = <AkitaFr as ZkBatchOpeningScheme>::verify_batch_zk(
        &verifier_setup,
        &mut transcript,
        &zk_statement,
        &dummy_proof,
    );
    assert!(matches!(verify_result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn akita_field_conversion_is_canonical() {
    let value = fr(42);
    let converted = to_akita_claim(value);
    let mut original_bytes = [0u8; Fr::NUM_BYTES];
    let mut converted_bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut original_bytes);
    converted.to_bytes_le(&mut converted_bytes);
    assert_eq!(converted, value);
    assert_eq!(converted_bytes, original_bytes);
}
