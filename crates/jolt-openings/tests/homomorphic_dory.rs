#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]

use jolt_crypto::Commitment;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch, OpeningsError,
    VerifierOpeningClaim, ZkBatchOpeningScheme, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type HomomorphicDoryBatch = HomomorphicBatch<DoryScheme>;
type DoryOutput = <DoryScheme as Commitment>::Output;
type DoryOpeningHint = <DoryScheme as CommitmentScheme>::OpeningHint;
type ClearClaim = VerifierOpeningClaim<Fr, DoryOutput>;

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
) -> (Vec<ClearClaim>, Vec<DoryOpeningHint>) {
    let mut claims = Vec::with_capacity(polynomials.len());
    let mut hints = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        let (commitment, hint) = DoryScheme::commit(polynomial, setup);
        claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
        });
        hints.push(hint);
    }
    (claims, hints)
}

fn sources(polynomials: &[Polynomial<Fr>]) -> Vec<&dyn MultilinearPoly<Fr>> {
    polynomials
        .iter()
        .map(|polynomial| polynomial as &dyn MultilinearPoly<Fr>)
        .collect()
}

#[test]
fn dory_homomorphic_batch_roundtrip_clear_many_polynomials() {
    let (polynomials, point) = homomorphic_polynomials(5, 4, 0x51_d0_42);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let verifier_setup = DoryScheme::setup_verifier(point.len());
    let (claims, hints) = clear_claims(&polynomials, &point, &prover_setup);

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
    let (claims, hints) = clear_claims(&polynomials, &point, &prover_setup);

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
    let (mut claims, hints) = clear_claims(&polynomials, &point, &prover_setup);
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
fn dory_homomorphic_batch_rejects_wrong_witness_dimension() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x71_00_02);
    let prover_setup = DoryScheme::setup_prover(point.len());
    let (claims, hints) = clear_claims(&polynomials, &point, &prover_setup);
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
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup);
        commitments.push(commitment);
        hints.push(hint);
        evaluations.push(polynomial.evaluate(&point));
    }

    let mut prover_transcript = Blake2bTranscript::new(b"dory-batch-zk");
    let (proof, hiding_commitment, _blind) =
        <HomomorphicDoryBatch as ZkBatchOpeningScheme>::prove_batch_zk(
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
    let mut hints = Vec::with_capacity(polynomials.len());
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for polynomial in &polynomials {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(polynomial, &prover_setup);
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
