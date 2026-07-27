#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]

use jolt_crypto::Bn254;
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{BatchOpeningScheme, HomomorphicBatch, OpeningsError};
use jolt_poly::{Point, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};

#[path = "support/common.rs"]
pub mod common;

use common::{clear_claims, fr, homomorphic_polynomials, kzg_setup, sources};

type KzgPCS = HyperKZGScheme<Bn254>;
type HomomorphicKzgBatch = HomomorphicBatch<KzgPCS>;

#[test]
fn hyperkzg_homomorphic_batch_roundtrip_clear_many_polynomials() {
    let (polynomials, point) = homomorphic_polynomials(5, 4, 0x70_55_1e);
    let (prover_setup, verifier_setup) = kzg_setup(point.len());
    let (claims, hints) = clear_claims::<KzgPCS>(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-batch");
    let proof = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        sources(&polynomials),
        hints,
        &mut prover_transcript,
    )
    .expect("HyperKZG homomorphic batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-batch");
    <HomomorphicKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &claims,
        &proof,
        &mut verifier_transcript,
    )
    .expect("HyperKZG homomorphic batch proof should verify");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_tampered_value() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x72_00_00);
    let (prover_setup, verifier_setup) = kzg_setup(point.len());
    let (claims, hints) = clear_claims::<KzgPCS>(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-batch-tamper");
    let proof = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        sources(&polynomials),
        hints,
        &mut prover_transcript,
    )
    .expect("HyperKZG homomorphic batch proof should be produced");

    let mut tampered = claims;
    tampered[0].evaluation.value += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-batch-tamper");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &tampered,
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered HyperKZG batch value should fail");
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_mismatched_point() {
    let (polynomials, point) = homomorphic_polynomials(4, 3, 0x72_00_01);
    let (prover_setup, _) = kzg_setup(point.len());
    let (mut claims, hints) = clear_claims::<KzgPCS>(&polynomials, &point, &prover_setup);
    claims[2].evaluation.point = Point::new(vec![fr(2), fr(3), fr(5)]);

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-batch-point-mismatch");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        sources(&polynomials),
        hints,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_witness_count_mismatch() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x72_00_02);
    let (prover_setup, _) = kzg_setup(point.len());
    let (claims, mut hints) = clear_claims::<KzgPCS>(&polynomials, &point, &prover_setup);
    let mut polynomial_sources = sources(&polynomials);
    let _dropped = polynomial_sources.pop();
    let _dropped_hint = hints.pop();

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-batch-witness-count");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        polynomial_sources,
        hints,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_wrong_witness_dimension() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x72_00_03);
    let (prover_setup, _) = kzg_setup(point.len());
    let (claims, hints) = clear_claims::<KzgPCS>(&polynomials, &point, &prover_setup);
    let wrong_witness = Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)]);
    let mut polynomial_sources = sources(&polynomials);
    polynomial_sources[1] = &wrong_witness;

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-batch-witness-dim");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        polynomial_sources,
        hints,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
