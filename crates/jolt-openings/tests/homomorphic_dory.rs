#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests assert successful proof paths"
)]

use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field, Fr};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch, OpeningsError,
    VerifierOpeningClaim, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};

#[path = "support/common.rs"]
pub mod common;

use common::{clear_claims, fr, homomorphic_polynomials, sources};

type HomomorphicDoryBatch = HomomorphicBatch<DoryScheme>;

#[test]
fn dory_homomorphic_batch_roundtrip_clear_many_polynomials() {
    let (polynomials, point) = homomorphic_polynomials(5, 4, 0x51_d0_42);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let verifier_setup = DoryScheme::setup_verifier(point.len());
    let (claims, hints) = clear_claims::<DoryScheme>(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-batch");
    let proof = <HomomorphicDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        sources(&polynomials),
        hints,
        &mut prover_transcript,
    )
    .expect("Dory homomorphic batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-batch");
    <HomomorphicDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &claims,
        &proof,
        &mut verifier_transcript,
    )
    .expect("Dory homomorphic batch proof should verify");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn dory_homomorphic_batch_rejects_tampered_value() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x71_00_00);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let verifier_setup = DoryScheme::setup_verifier(point.len());
    let (claims, hints) = clear_claims::<DoryScheme>(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-batch-tamper");
    let proof = <HomomorphicDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        sources(&polynomials),
        hints,
        &mut prover_transcript,
    )
    .expect("Dory homomorphic batch proof should be produced");

    let mut tampered = claims;
    tampered[1].evaluation.value += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-batch-tamper");
    let result = <HomomorphicDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &tampered,
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered Dory batch value should fail");
}

#[test]
fn dory_homomorphic_batch_rejects_mismatched_points() {
    let (polynomials, point) = homomorphic_polynomials(4, 3, 0x71_00_01);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let (mut claims, hints) = clear_claims::<DoryScheme>(&polynomials, &point, &prover_setup);
    claims[2].evaluation.point = Point::new(vec![fr(2), fr(3), fr(5)]);

    let mut transcript = Blake2bTranscript::new(b"dory-batch-point-mismatch");
    let result = <HomomorphicDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        sources(&polynomials),
        hints,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_homomorphic_batch_rejects_witness_count_mismatch() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x71_00_05);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let (claims, mut hints) = clear_claims::<DoryScheme>(&polynomials, &point, &prover_setup);
    let mut polynomial_sources = sources(&polynomials);
    let _dropped = polynomial_sources.pop();
    let _dropped_hint = hints.pop();

    let mut transcript = Blake2bTranscript::new(b"dory-batch-witness-count");
    let result = <HomomorphicDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        polynomial_sources,
        hints,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_homomorphic_batch_rejects_wrong_witness_dimension() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x71_00_02);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let (claims, hints) = clear_claims::<DoryScheme>(&polynomials, &point, &prover_setup);
    let wrong_witness = Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)]);
    let mut polynomial_sources = sources(&polynomials);
    polynomial_sources[0] = &wrong_witness;

    let mut transcript = Blake2bTranscript::new(b"dory-batch-witness-dim");
    let result = <HomomorphicDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        polynomial_sources,
        hints,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_homomorphic_zk_batch_roundtrip() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x71_00_03);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let verifier_setup = DoryScheme::setup_verifier(point.len());
    let mut commitments = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for polynomial in &polynomials {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup).unwrap();
        commitments.push(commitment);
        hints.push(hint);
        evaluations.push(polynomial.evaluate(&point));
    }

    let mut prover_transcript = Blake2bTranscript::new(b"dory-batch-zk");
    let opening = <HomomorphicDoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        point.clone(),
        commitments.clone(),
        sources(&polynomials),
        hints,
        evaluations,
        &mut prover_transcript,
    )
    .expect("Dory ZK homomorphic batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-batch-zk");
    let verifier_hiding = <HomomorphicDoryBatch as ZkBatchOpeningScheme>::verify_batch_zk(
        &verifier_setup,
        point,
        commitments,
        &opening.proof,
        &mut verifier_transcript,
    )
    .expect("Dory ZK homomorphic batch proof should verify");

    assert_eq!(opening.hiding_commitment, verifier_hiding);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

struct ZkBatchFixture {
    point: Point<HIGH_TO_LOW, Fr>,
    commitments: Vec<DoryCommitment>,
    verifier_setup: DoryVerifierSetup,
    proof: DoryProof,
}

/// Produces an honest ZK batch proof over two random 2-variable polynomials,
/// bound to a caller-chosen transcript label so each test can replay the
/// exact prover transcript.
fn zk_batch_fixture(seed: u64, label: &'static [u8]) -> ZkBatchFixture {
    let (polynomials, point) = homomorphic_polynomials(2, 2, seed);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let verifier_setup = DoryScheme::setup_verifier(point.len());
    let mut commitments = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for polynomial in &polynomials {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup).unwrap();
        commitments.push(commitment);
        hints.push(hint);
        evaluations.push(polynomial.evaluate(&point));
    }

    let mut transcript = Blake2bTranscript::new(label);
    let opening = <HomomorphicDoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        point.clone(),
        commitments.clone(),
        sources(&polynomials),
        hints,
        evaluations,
        &mut transcript,
    )
    .expect("honest Dory ZK batch proof should be produced");

    ZkBatchFixture {
        point,
        commitments,
        verifier_setup,
        proof: opening.proof,
    }
}

fn verify_zk_fixture(
    fixture: &ZkBatchFixture,
    label: &'static [u8],
    point: Point<HIGH_TO_LOW, Fr>,
    commitments: Vec<DoryCommitment>,
    proof: &DoryProof,
) -> Result<<DoryScheme as ZkOpeningScheme>::HidingCommitment, OpeningsError> {
    let mut transcript = Blake2bTranscript::new(label);
    <HomomorphicDoryBatch as ZkBatchOpeningScheme>::verify_batch_zk(
        &fixture.verifier_setup,
        point,
        commitments,
        proof,
        &mut transcript,
    )
}

#[test]
fn dory_homomorphic_zk_batch_rejects_tampered_commitment() {
    let label = b"dory-batch-zk-commitment-tamper";
    let fixture = zk_batch_fixture(0x71_01_00, label);

    // Swapping the two commitments keeps them individually well-formed but
    // reassigns each to the wrong polynomial, so the verifier's joint
    // commitment no longer matches the proven joint polynomial.
    let mut tampered = fixture.commitments.clone();
    assert_ne!(
        tampered[0], tampered[1],
        "distinct commitments are required for the swap to be a tamper"
    );
    tampered.swap(0, 1);

    let result = verify_zk_fixture(
        &fixture,
        label,
        fixture.point.clone(),
        tampered,
        &fixture.proof,
    );
    assert!(
        matches!(result, Err(OpeningsError::VerificationFailed)),
        "swapped commitments must fail verification: {result:?}"
    );

    // Control: the untampered statement verifies under the same label.
    let _hiding = verify_zk_fixture(
        &fixture,
        label,
        fixture.point.clone(),
        fixture.commitments.clone(),
        &fixture.proof,
    )
    .expect("control: the untampered ZK batch proof must verify");
}

#[test]
fn dory_homomorphic_zk_batch_rejects_wrong_opening_point() {
    let label = b"dory-batch-zk-point-tamper";
    let fixture = zk_batch_fixture(0x71_01_01, label);

    let mut coordinates = fixture.point.as_slice().to_vec();
    coordinates[0] += fr(1);
    let wrong_point = Point::new(coordinates);

    let result = verify_zk_fixture(
        &fixture,
        label,
        wrong_point,
        fixture.commitments.clone(),
        &fixture.proof,
    );
    assert!(
        matches!(result, Err(OpeningsError::VerificationFailed)),
        "a shifted opening point must fail verification: {result:?}"
    );

    // Control: the point the proof was produced for verifies.
    let _hiding = verify_zk_fixture(
        &fixture,
        label,
        fixture.point.clone(),
        fixture.commitments.clone(),
        &fixture.proof,
    )
    .expect("control: the original opening point must verify");
}

#[test]
fn dory_homomorphic_zk_batch_rejects_tampered_hiding_commitment() {
    let label = b"dory-batch-zk-ycom-tamper";
    let fixture = zk_batch_fixture(0x71_01_02, label);
    let donor = zk_batch_fixture(0x71_01_03, b"dory-batch-zk-ycom-donor");

    // Graft a well-formed but wrong evaluation commitment (y_com) from an
    // unrelated proof: the value the proof binds no longer matches the
    // proven evaluation.
    let mut grafted = fixture.proof.clone();
    assert_ne!(
        grafted.0.y_com, donor.proof.0.y_com,
        "the donor proof must supply a different hiding commitment"
    );
    grafted.0.y_com = donor.proof.0.y_com;

    let result = verify_zk_fixture(
        &fixture,
        label,
        fixture.point.clone(),
        fixture.commitments.clone(),
        &grafted,
    );
    assert!(
        matches!(result, Err(OpeningsError::VerificationFailed)),
        "a grafted hiding commitment must fail verification: {result:?}"
    );

    // Stripping the hiding commitment entirely must also be rejected rather
    // than falling back to a non-hiding verification path.
    let mut stripped = fixture.proof.clone();
    stripped.0.y_com = None;
    let result = verify_zk_fixture(
        &fixture,
        label,
        fixture.point.clone(),
        fixture.commitments.clone(),
        &stripped,
    );
    assert!(
        matches!(result, Err(OpeningsError::VerificationFailed)),
        "a missing hiding commitment must fail verification: {result:?}"
    );

    // Control: the intact proof verifies.
    let _hiding = verify_zk_fixture(
        &fixture,
        label,
        fixture.point.clone(),
        fixture.commitments.clone(),
        &fixture.proof,
    )
    .expect("control: the intact ZK batch proof must verify");
}

#[test]
fn dory_homomorphic_zk_batch_rejects_witness_count_mismatch() {
    let (polynomials, point) = homomorphic_polynomials(2, 2, 0x71_00_04);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let mut commitments = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for polynomial in &polynomials {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup).unwrap();
        commitments.push(commitment);
        hints.push(hint);
        evaluations.push(polynomial.evaluate(&point));
    }
    let mut polynomial_sources = sources(&polynomials);
    let _dropped = polynomial_sources.pop();
    let _dropped_hint = hints.pop();
    let _dropped_eval = evaluations.pop();

    let mut transcript = Blake2bTranscript::new(b"dory-batch-zk-witness-count");
    let result = <HomomorphicDoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        point,
        commitments,
        polynomial_sources,
        hints,
        evaluations,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

/// The batching challenge must depend on the commitments, not on the claimed
/// values alone.
///
/// If it does not, a prover can draw the challenge first and then choose a
/// commitment that makes the random linear combination consistent while the
/// individual claims are false. Here `p2` is the constant `gamma^-1`, so
/// `gamma * p2` is the constant one and the joint claim absorbs the `+1` lie
/// in claim 1. Both claims in the batch are false, and the batch verifier must
/// still reject.
#[test]
fn dory_homomorphic_batch_binds_commitments_to_the_batching_challenge() {
    let num_vars = 3;
    let (polynomials, point) = homomorphic_polynomials(1, num_vars, 0x71_00_05);
    let p1 = &polynomials[0];

    // Claim 1 overstates p1 by one; claim 2 understates p2 to zero.
    let v1 = p1.evaluate(&point) + fr(1);
    let v2 = fr(0);

    // A challenge drawn from the two values alone is one the prover can know
    // before it picks the second commitment.
    let mut replay = Blake2bTranscript::new(b"dory-batch-commitment-binding");
    replay.append(&LabelWithCount(b"rlc_claims", 2));
    v1.append_to_transcript(&mut replay);
    v2.append_to_transcript(&mut replay);
    let gamma: Fr = replay.challenge_scalar_powers(2)[1];
    let p2 = Polynomial::new(vec![
        gamma.inverse().expect("gamma is non-zero");
        1 << num_vars
    ]);

    let prover_setup = DoryScheme::setup_prover(num_vars);
    let verifier_setup = DoryScheme::setup_verifier(num_vars);
    let (c1, hint1) = DoryScheme::commit(p1, &prover_setup).unwrap();
    let (c2, hint2) = DoryScheme::commit(&p2, &prover_setup).unwrap();
    let claims = vec![
        VerifierOpeningClaim {
            commitment: c1,
            evaluation: EvaluationClaim::new(point.clone(), v1),
        },
        VerifierOpeningClaim {
            commitment: c2,
            evaluation: EvaluationClaim::new(point, v2),
        },
    ];

    let mut prover_transcript = Blake2bTranscript::new(b"dory-batch-commitment-binding");
    let proof = <HomomorphicDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        vec![
            p1 as &dyn MultilinearPoly<Fr>,
            &p2 as &dyn MultilinearPoly<Fr>,
        ],
        vec![hint1, hint2],
        &mut prover_transcript,
    )
    .expect("the combined polynomial really does evaluate to the joint claim");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-batch-commitment-binding");
    let result = <HomomorphicDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &claims,
        &proof,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "batch verifier accepted two false claims because the batching challenge \
         was not bound to the commitments"
    );
}
