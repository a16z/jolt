#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]

use jolt_crypto::Commitment;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch, OpeningsError,
    VerifierOpeningClaim, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type HomBatch = HomomorphicBatch<DoryScheme>;
type DoryOutput = <DoryScheme as Commitment>::Output;
type DoryOpeningHint = <DoryScheme as CommitmentScheme>::OpeningHint;
type ClearClaim = VerifierOpeningClaim<Fr, DoryOutput>;
type ClearWitness = (Polynomial<Fr>, DoryOpeningHint);

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn random_point(num_vars: usize, rng: &mut ChaCha20Rng) -> Point<HIGH_TO_LOW, Fr> {
    Point::new((0..num_vars).map(|_| Fr::random(rng)).collect::<Vec<_>>())
}

fn homomorphic_polynomials(
    count: usize,
    num_vars: usize,
    seed: u64,
) -> (Vec<Polynomial<Fr>>, Point<HIGH_TO_LOW, Fr>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let polynomials = (0..count)
        .map(|_| Polynomial::<Fr>::random(num_vars, &mut rng))
        .collect();
    let point = random_point(num_vars, &mut rng);
    (polynomials, point)
}

fn clear_claims(
    polynomials: &[Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (Vec<ClearClaim>, Vec<ClearWitness>) {
    let mut claims = Vec::with_capacity(polynomials.len());
    let mut witness = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        let (commitment, hint) = DoryScheme::commit(polynomial, setup);
        claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
        });
        witness.push((polynomial.clone(), hint));
    }
    (claims, witness)
}

#[test]
fn dory_homomorphic_batch_roundtrip_clear_many_polynomials() {
    let (polynomials, point) = homomorphic_polynomials(5, 4, 0x51_d0_42);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let verifier_setup = DoryScheme::setup_verifier(point.len());
    let (claims, witness) = clear_claims(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-hom-batch");
    let proof = <HomBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        witness,
        &mut prover_transcript,
    )
    .expect("Dory homomorphic batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-hom-batch");
    <HomBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        claims,
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
    let (claims, witness) = clear_claims(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-hom-tamper");
    let proof = <HomBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        witness,
        &mut prover_transcript,
    )
    .expect("Dory homomorphic batch proof should be produced");

    let mut tampered = claims;
    tampered[1].evaluation.value += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-hom-tamper");
    let result = <HomBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        tampered,
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered Dory batch value should fail");
}

#[test]
fn dory_homomorphic_batch_rejects_mismatched_points() {
    let (polynomials, point) = homomorphic_polynomials(4, 3, 0x71_00_01);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let (mut claims, witness) = clear_claims(&polynomials, &point, &prover_setup);
    claims[2].evaluation.point = Point::new(vec![fr(2), fr(3), fr(5)]);

    let mut transcript = Blake2bTranscript::new(b"dory-hom-point-mismatch");
    let result = <HomBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        witness,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_homomorphic_batch_rejects_wrong_witness_dimension() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x71_00_02);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let (claims, mut witness) = clear_claims(&polynomials, &point, &prover_setup);
    witness[0].0 = Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)]);

    let mut transcript = Blake2bTranscript::new(b"dory-hom-witness-dim");
    let result = <HomBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        witness,
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
    let mut witness = Vec::with_capacity(polynomials.len());
    for polynomial in &polynomials {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup);
        commitments.push(commitment);
        witness.push((polynomial.clone(), hint, polynomial.evaluate(&point)));
    }

    let mut prover_transcript = Blake2bTranscript::new(b"dory-hom-zk-batch");
    let (proof, hiding_commitment, _blind) = <HomBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        point.clone(),
        commitments.clone(),
        witness,
        &mut prover_transcript,
    )
    .expect("Dory ZK homomorphic batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-hom-zk-batch");
    let verifier_hiding = <HomBatch as ZkBatchOpeningScheme>::verify_batch_zk(
        &verifier_setup,
        point,
        commitments,
        &proof,
        &mut verifier_transcript,
    )
    .expect("Dory ZK homomorphic batch proof should verify");

    assert_eq!(hiding_commitment, verifier_hiding);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn dory_homomorphic_zk_batch_rejects_witness_count_mismatch() {
    let (polynomials, point) = homomorphic_polynomials(2, 2, 0x71_00_04);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let mut commitments = Vec::with_capacity(polynomials.len());
    let mut witness = Vec::with_capacity(polynomials.len());
    for polynomial in &polynomials {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup);
        commitments.push(commitment);
        witness.push((polynomial.clone(), hint, polynomial.evaluate(&point)));
    }
    let _dropped = witness.pop();

    let mut transcript = Blake2bTranscript::new(b"dory-hom-zk-witness-count");
    let result = <HomBatch as ZkBatchOpeningScheme>::prove_batch_zk(
        &prover_setup,
        point,
        commitments,
        witness,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
