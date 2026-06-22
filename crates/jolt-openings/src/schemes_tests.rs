#![expect(
    clippy::expect_used,
    reason = "tests assert successful batch proof results"
)]

use super::*;
use crate::mock::MockCommitmentScheme;
use jolt_field::{Fr, FromPrimitiveInt, Invertible};
use jolt_poly::Polynomial;
use jolt_transcript::Blake2bTranscript;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpeningId {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationId {
    First,
    Second,
}

fn packed_term<F>(coefficient: F) -> PackedLinearTerm<F> {
    PackedLinearTerm::new(coefficient, PackedFamilyRef::new(0x6a6f_6c74, 1, 0), 0, 0)
}

fn packed_term_at<F>(coefficient: F, symbol: usize) -> PackedLinearTerm<F> {
    PackedLinearTerm::new(
        coefficient,
        PackedFamilyRef::new(0x6a6f_6c74, 1, 0),
        0,
        symbol,
    )
}

#[test]
fn batch_opening_statement_preserves_claim_metadata() {
    let statement = BatchOpeningStatement {
        logical_point: vec![3_u64, 5],
        pcs_point: vec![5_u64, 3],
        layout_digest: [7; 32],
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: 11_u64,
                claim: 13,
                view: PhysicalView::Direct,
                scale: 17,
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Second,
                commitment: 19_u64,
                claim: 23,
                view: PhysicalView::PackedLinear {
                    layout_digest: [23; 32],
                    terms: vec![packed_term(29), packed_term_at(31, 1)],
                },
                scale: 37,
            },
        ],
    };

    assert_eq!(statement.logical_point, vec![3, 5]);
    assert_eq!(statement.pcs_point, vec![5, 3]);
    assert_eq!(statement.layout_digest, [7; 32]);
    assert_eq!(statement.claims[0].id, OpeningId::A);
    assert_eq!(statement.claims[0].relation, RelationId::First);
    assert_eq!(statement.claims[0].claim, 13);
    assert_eq!(statement.claims[1].id, OpeningId::B);
    assert_eq!(statement.claims[1].relation, RelationId::Second);
    assert_eq!(statement.claims[1].claim, 23);
}

#[test]
fn zk_batch_opening_statement_can_omit_claim_payload() {
    let statement: BatchOpeningStatement<_, _, _, _, ()> = BatchOpeningStatement {
        logical_point: vec![1_u64],
        pcs_point: vec![1_u64],
        layout_digest: [0; 32],
        claims: vec![BatchOpeningClaim {
            id: OpeningId::A,
            relation: RelationId::First,
            commitment: 2_u64,
            claim: (),
            view: PhysicalView::Direct,
            scale: 1,
        }],
    };

    let _: () = statement.claims[0].claim;
}

#[test]
fn physical_view_records_packed_layout_and_terms() {
    let view = PhysicalView::PackedLinear {
        layout_digest: [41; 32],
        terms: vec![packed_term(43_u64), packed_term_at(47, 1)],
    };

    assert!(matches!(view, PhysicalView::PackedLinear { .. }));
    if let PhysicalView::PackedLinear {
        layout_digest,
        terms,
    } = view
    {
        assert_eq!(layout_digest, [41; 32]);
        assert_eq!(terms[0].coefficient, 43);
        assert_eq!(terms[1].coefficient, 47);
        assert_eq!(terms[1].symbol, 1);
    }
}

type MockPCS = MockCommitmentScheme<Fr>;
fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn batch_polynomials() -> (Vec<Polynomial<Fr>>, Vec<Fr>) {
    let polynomials = vec![
        Polynomial::new((0..8).map(|value| fr(value + 1)).collect()),
        Polynomial::new((0..8).map(|value| fr(17 + 2 * value)).collect()),
    ];
    let point = vec![fr(2), fr(3), fr(5)];
    (polynomials, point)
}

fn clear_batch_statement(
    polynomials: &[Polynomial<Fr>],
    point: &[Fr],
) -> BatchOpeningStatement<Fr, <MockPCS as Commitment>::Output, OpeningId, RelationId> {
    let commitments = polynomials
        .iter()
        .map(|polynomial| MockPCS::commit(polynomial.evaluations(), &()).0)
        .collect::<Vec<_>>();
    let first_scale = fr(2);
    let second_scale = fr(5);
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: [9; 32],
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: commitments[0].clone(),
                claim: polynomials[0].evaluate(point)
                    * first_scale.inverse().expect("scale is nonzero"),
                view: PhysicalView::Direct,
                scale: first_scale,
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Second,
                commitment: commitments[1].clone(),
                claim: polynomials[1].evaluate(point)
                    * second_scale.inverse().expect("scale is nonzero"),
                view: PhysicalView::Direct,
                scale: second_scale,
            },
        ],
    }
}

#[test]
fn homomorphic_batch_opening_roundtrip_clear() {
    let (polynomials, point) = batch_polynomials();
    let statement = clear_batch_statement(&polynomials, &point);
    let hints = vec![(); polynomials.len()];

    let mut prover_transcript = Blake2bTranscript::new(b"batch-clear");
    let proof = <MockPCS as BatchOpeningScheme>::prove_batch(
        &(),
        &mut prover_transcript,
        &statement,
        &polynomials,
        hints,
    )
    .expect("batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear");
    let result = <MockPCS as BatchOpeningScheme>::verify_batch(
        &(),
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("batch proof should verify");

    assert_eq!(result.coefficients.len(), statement.claims.len());
    assert_eq!(
        result.reduced_opening,
        result
            .coefficients
            .iter()
            .zip(&statement.claims)
            .fold(Fr::from_u64(0), |acc, (coefficient, claim)| {
                acc + *coefficient * claim.claim
            })
    );
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn homomorphic_batch_opening_rejects_tampered_clear_claim() {
    let (polynomials, point) = batch_polynomials();
    let statement = clear_batch_statement(&polynomials, &point);

    let mut prover_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
    let proof = <MockPCS as BatchOpeningScheme>::prove_batch(
        &(),
        &mut prover_transcript,
        &statement,
        &polynomials,
        vec![(); polynomials.len()],
    )
    .expect("batch proof should be produced");

    let mut tampered = statement.clone();
    tampered.claims[1].claim += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
    let result = <MockPCS as BatchOpeningScheme>::verify_batch(
        &(),
        &mut verifier_transcript,
        &tampered,
        &proof,
    );
    assert!(result.is_err(), "tampered claim should fail");
}

#[test]
fn homomorphic_batch_opening_rejects_packed_view() {
    let (polynomials, point) = batch_polynomials();
    let mut statement = clear_batch_statement(&polynomials, &point);
    statement.claims[0].view = PhysicalView::PackedLinear {
        layout_digest: [3; 32],
        terms: vec![packed_term(fr(1)), packed_term_at(fr(2), 1)],
    };

    let mut transcript = Blake2bTranscript::new(b"batch-packed");
    let result = <MockPCS as BatchOpeningScheme>::prove_batch(
        &(),
        &mut transcript,
        &statement,
        &polynomials,
        vec![(); polynomials.len()],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn homomorphic_batch_opening_rejects_mismatched_witness_count() {
    let (polynomials, point) = batch_polynomials();
    let statement = clear_batch_statement(&polynomials, &point);

    let mut transcript = Blake2bTranscript::new(b"batch-mismatch");
    let result = <MockPCS as BatchOpeningScheme>::prove_batch(
        &(),
        &mut transcript,
        &statement,
        &polynomials[..1],
        vec![()],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

fn zk_batch_statement(
    polynomials: &[Polynomial<Fr>],
    point: &[Fr],
) -> BatchOpeningStatement<Fr, <MockPCS as Commitment>::Output, OpeningId, RelationId, ()> {
    let commitments = polynomials
        .iter()
        .map(|polynomial| <MockPCS as ZkOpeningScheme>::commit_zk(polynomial.evaluations(), &()).0)
        .collect::<Vec<_>>();
    BatchOpeningStatement {
        logical_point: point.to_vec(),
        pcs_point: point.to_vec(),
        layout_digest: [10; 32],
        claims: vec![
            BatchOpeningClaim {
                id: OpeningId::A,
                relation: RelationId::First,
                commitment: commitments[0].clone(),
                claim: (),
                view: PhysicalView::Direct,
                scale: fr(3),
            },
            BatchOpeningClaim {
                id: OpeningId::B,
                relation: RelationId::Second,
                commitment: commitments[1].clone(),
                claim: (),
                view: PhysicalView::Direct,
                scale: fr(7),
            },
        ],
    }
}

#[test]
fn homomorphic_batch_opening_roundtrip_zk() {
    let (polynomials, point) = batch_polynomials();
    let statement = zk_batch_statement(&polynomials, &point);
    let evals = polynomials
        .iter()
        .map(|polynomial| polynomial.evaluate(&point))
        .collect::<Vec<_>>();

    let mut prover_transcript = Blake2bTranscript::new(b"batch-zk");
    let (proof, hiding_commitment, _blind) = <MockPCS as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        &mut prover_transcript,
        &statement,
        &evals,
        &polynomials,
        vec![(); polynomials.len()],
    )
    .expect("ZK batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-zk");
    let result = <MockPCS as ZkBatchOpeningScheme>::verify_batch_zk(
        &(),
        &mut verifier_transcript,
        &statement,
        &proof,
    )
    .expect("ZK batch proof should verify");

    assert_eq!(result.reduced_opening, hiding_commitment);
    assert_eq!(result.coefficients.len(), statement.claims.len());
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn homomorphic_zk_batch_opening_rejects_eval_count_mismatch() {
    let (polynomials, point) = batch_polynomials();
    let statement = zk_batch_statement(&polynomials, &point);
    let evals = [polynomials[0].evaluate(&point)];

    let mut transcript = Blake2bTranscript::new(b"batch-zk-mismatch");
    let result = <MockPCS as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        &mut transcript,
        &statement,
        &evals,
        &polynomials,
        vec![(); polynomials.len()],
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
