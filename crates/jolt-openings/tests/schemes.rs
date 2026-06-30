#![expect(
    clippy::expect_used,
    reason = "tests assert successful batch proof results"
)]

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch,
    HomomorphicBatchWitness, HomomorphicZkBatchWitness, OpeningsError, VerifierOpeningClaim,
    ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};

#[path = "support/mock.rs"]
mod mock;

use mock::{MockCommitment, MockCommitmentScheme};

type MockPCS = MockCommitmentScheme<Fr>;
type HomomorphicTestBatch = HomomorphicBatch<MockPCS>;

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn batch_polynomials() -> (Vec<Polynomial<Fr>>, Point<HIGH_TO_LOW, Fr>) {
    let polynomials = vec![
        Polynomial::new((0..8).map(|value| fr(value + 1)).collect()),
        Polynomial::new((0..8).map(|value| fr(17 + 2 * value)).collect()),
    ];
    let point = Point::new(vec![fr(2), fr(3), fr(5)]);
    (polynomials, point)
}

fn clear_claims(
    polynomials: &[Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
) -> Vec<VerifierOpeningClaim<Fr, MockCommitment<Fr>>> {
    polynomials
        .iter()
        .map(|polynomial| {
            let (commitment, ()) = MockPCS::commit(polynomial, &());
            VerifierOpeningClaim {
                commitment,
                evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
            }
        })
        .collect()
}

fn clear_witness(polynomials: &[Polynomial<Fr>]) -> HomomorphicBatchWitness<'_, Fr, ()> {
    polynomials
        .iter()
        .map(|polynomial| (polynomial as &dyn MultilinearPoly<Fr>, ()))
        .collect()
}

#[test]
fn homomorphic_batch_opening_roundtrip_clear() {
    let (polynomials, point) = batch_polynomials();
    let claims = clear_claims(&polynomials, &point);

    let mut prover_transcript = Blake2bTranscript::new(b"batch-clear");
    let proof = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims.clone(),
        clear_witness(&polynomials),
        &mut prover_transcript,
    )
    .expect("batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear");
    <HomomorphicTestBatch as BatchOpeningScheme>::verify_batch(
        &(),
        claims,
        &proof,
        &mut verifier_transcript,
    )
    .expect("batch proof should verify");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn homomorphic_batch_opening_rejects_tampered_clear_claim() {
    let (polynomials, point) = batch_polynomials();
    let claims = clear_claims(&polynomials, &point);

    let mut prover_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
    let proof = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims.clone(),
        clear_witness(&polynomials),
        &mut prover_transcript,
    )
    .expect("batch proof should be produced");

    let mut tampered = claims;
    tampered[1].evaluation.value += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-clear-tampered");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::verify_batch(
        &(),
        tampered,
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered claim should fail");
}

#[test]
fn homomorphic_batch_opening_rejects_empty_claims() {
    let mut transcript = Blake2bTranscript::new(b"batch-empty");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        Vec::new(),
        Vec::new(),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn homomorphic_batch_opening_rejects_mismatched_points() {
    let (polynomials, point) = batch_polynomials();
    let mut claims = clear_claims(&polynomials, &point);
    claims[1].evaluation.point = Point::new(vec![fr(8), fr(13), fr(21)]);

    let mut transcript = Blake2bTranscript::new(b"batch-point-mismatch");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims,
        clear_witness(&polynomials),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn homomorphic_batch_opening_rejects_mismatched_witness_count() {
    let (polynomials, point) = batch_polynomials();
    let claims = clear_claims(&polynomials, &point);

    let mut transcript = Blake2bTranscript::new(b"batch-mismatch");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims,
        clear_witness(&polynomials[..1]),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

fn zk_commitments(polynomials: &[Polynomial<Fr>]) -> Vec<MockCommitment<Fr>> {
    polynomials
        .iter()
        .map(|polynomial| <MockPCS as ZkOpeningScheme>::commit_zk(polynomial, &()).0)
        .collect()
}

fn zk_witness<'a>(
    polynomials: &'a [Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
) -> HomomorphicZkBatchWitness<'a, Fr, ()> {
    polynomials
        .iter()
        .map(|polynomial| {
            let eval = polynomial.evaluate(point);
            (polynomial as &dyn MultilinearPoly<Fr>, (), eval)
        })
        .collect()
}

#[test]
fn homomorphic_batch_opening_roundtrip_zk() {
    let (polynomials, point) = batch_polynomials();
    let commitments = zk_commitments(&polynomials);

    let mut prover_transcript = Blake2bTranscript::new(b"batch-zk");
    let (proof, hiding_commitment, _blind) =
        <HomomorphicTestBatch as ZkBatchOpeningScheme>::prove_batch_zk(
            &(),
            point.clone(),
            commitments.clone(),
            zk_witness(&polynomials, &point),
            &mut prover_transcript,
        )
        .expect("ZK batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"batch-zk");
    let verifier_hiding = <HomomorphicTestBatch as ZkBatchOpeningScheme>::verify_batch_zk(
        &(),
        point,
        commitments,
        &proof,
        &mut verifier_transcript,
    )
    .expect("ZK batch proof should verify");

    assert_eq!(verifier_hiding, hiding_commitment);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn homomorphic_zk_batch_opening_rejects_witness_count_mismatch() {
    let (polynomials, point) = batch_polynomials();
    let commitments = zk_commitments(&polynomials);

    let mut transcript = Blake2bTranscript::new(b"batch-zk-mismatch");
    let result = <HomomorphicTestBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        point.clone(),
        commitments,
        zk_witness(&polynomials[..1], &point),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
