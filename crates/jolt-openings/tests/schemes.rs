#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests assert successful batch proof results"
)]

use jolt_field::Fr;
use jolt_openings::{
    BatchOpeningScheme, HomomorphicBatch, OpeningsError, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{Point, Polynomial, HIGH_TO_LOW};

#[path = "support/common.rs"]
pub mod common;
#[path = "support/mock.rs"]
pub mod mock;

use common::{clear_claims, fingerprint, fr, prover, sources, verifier};
use mock::{MockCommitment, MockCommitmentScheme};

type MockPCS = MockCommitmentScheme<Fr>;
type HomomorphicTestBatch = HomomorphicBatch<MockPCS>;

fn batch_polynomials() -> (Vec<Polynomial<Fr>>, Point<HIGH_TO_LOW, Fr>) {
    let polynomials = vec![
        Polynomial::new((0..8).map(|value| fr(value + 1)).collect()),
        Polynomial::new((0..8).map(|value| fr(17 + 2 * value)).collect()),
    ];
    let point = Point::new(vec![fr(2), fr(3), fr(5)]);
    (polynomials, point)
}

fn unit_hints(count: usize) -> Vec<()> {
    vec![(); count]
}

#[test]
fn homomorphic_batch_opening_roundtrip_clear() {
    let (polynomials, point) = batch_polynomials();
    let (claims, _) = clear_claims::<MockPCS>(&polynomials, &point, &());

    let mut prover_transcript = prover(b"batch-clear");
    <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims.clone(),
        sources(&polynomials),
        unit_hints(polynomials.len()),
        &mut prover_transcript,
    )
    .expect("batch proof should be produced");

    let narg = prover_transcript.narg().to_vec();
    let mut verifier_transcript = verifier(b"batch-clear", &narg);
    <HomomorphicTestBatch as BatchOpeningScheme>::verify_batch(
        &(),
        &claims,
        &mut verifier_transcript,
    )
    .expect("batch proof should verify");

    assert_eq!(
        fingerprint(&prover_transcript),
        fingerprint(&verifier_transcript)
    );
    verifier_transcript.finish().expect("proof fully consumed");
}

#[test]
fn homomorphic_batch_opening_rejects_tampered_clear_claim() {
    let (polynomials, point) = batch_polynomials();
    let (claims, _) = clear_claims::<MockPCS>(&polynomials, &point, &());

    let mut prover_transcript = prover(b"batch-clear-tampered");
    <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims.clone(),
        sources(&polynomials),
        unit_hints(polynomials.len()),
        &mut prover_transcript,
    )
    .expect("batch proof should be produced");

    let mut tampered = claims;
    tampered[1].evaluation.value += fr(1);

    let narg = prover_transcript.finish();
    let mut verifier_transcript = verifier(b"batch-clear-tampered", &narg);
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::verify_batch(
        &(),
        &tampered,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered claim should fail");
}

#[test]
fn homomorphic_batch_opening_rejects_empty_claims() {
    let mut transcript = prover(b"batch-empty");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn homomorphic_batch_opening_rejects_mismatched_points() {
    let (polynomials, point) = batch_polynomials();
    let (mut claims, _) = clear_claims::<MockPCS>(&polynomials, &point, &());
    claims[1].evaluation.point = Point::new(vec![fr(8), fr(13), fr(21)]);

    let mut transcript = prover(b"batch-point-mismatch");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims,
        sources(&polynomials),
        unit_hints(polynomials.len()),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn homomorphic_batch_opening_rejects_mismatched_witness_count() {
    let (polynomials, point) = batch_polynomials();
    let (claims, _) = clear_claims::<MockPCS>(&polynomials, &point, &());

    let mut transcript = prover(b"batch-mismatch");
    let result = <HomomorphicTestBatch as BatchOpeningScheme>::prove_batch(
        &(),
        claims,
        sources(&polynomials[..1]),
        unit_hints(1),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

fn zk_commitments(polynomials: &[Polynomial<Fr>]) -> Vec<MockCommitment<Fr>> {
    polynomials
        .iter()
        .map(|polynomial| {
            <MockPCS as ZkOpeningScheme>::commit_zk(polynomial, &())
                .unwrap()
                .0
        })
        .collect()
}

fn evaluations(polynomials: &[Polynomial<Fr>], point: &Point<HIGH_TO_LOW, Fr>) -> Vec<Fr> {
    polynomials
        .iter()
        .map(|polynomial| polynomial.evaluate(point))
        .collect()
}

#[test]
fn homomorphic_batch_opening_roundtrip_zk() {
    let (polynomials, point) = batch_polynomials();
    let commitments = zk_commitments(&polynomials);

    let mut prover_transcript = prover(b"batch-zk");
    let opening = <HomomorphicTestBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        point.clone(),
        commitments.clone(),
        sources(&polynomials),
        unit_hints(polynomials.len()),
        evaluations(&polynomials, &point),
        &mut prover_transcript,
    )
    .expect("ZK batch proof should be produced");

    let narg = prover_transcript.narg().to_vec();
    let mut verifier_transcript = verifier(b"batch-zk", &narg);
    let verifier_hiding = <HomomorphicTestBatch as ZkBatchOpeningScheme>::verify_batch_zk(
        &(),
        point,
        commitments,
        &mut verifier_transcript,
    )
    .expect("ZK batch proof should verify");

    assert_eq!(verifier_hiding, opening.hiding_commitment);
    assert_eq!(
        fingerprint(&prover_transcript),
        fingerprint(&verifier_transcript)
    );
    verifier_transcript.finish().expect("proof fully consumed");
}

#[test]
fn homomorphic_zk_batch_opening_rejects_witness_count_mismatch() {
    let (polynomials, point) = batch_polynomials();
    let commitments = zk_commitments(&polynomials);

    let mut transcript = prover(b"batch-zk-mismatch");
    let result = <HomomorphicTestBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &(),
        point.clone(),
        commitments,
        sources(&polynomials[..1]),
        unit_hints(1),
        evaluations(&polynomials[..1], &point),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
