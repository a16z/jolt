#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]

use jolt_crypto::{Bn254, Commitment};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch,
    HomomorphicBatchWitness, OpeningsError, VerifierOpeningClaim,
};
use jolt_poly::{MultilinearPoly, Point, Polynomial, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;
type HomomorphicKzgBatch = HomomorphicBatch<KzgPCS>;
type KzgOutput = <KzgPCS as Commitment>::Output;
type KzgOpeningHint = <KzgPCS as CommitmentScheme>::OpeningHint;
type ClearClaim = VerifierOpeningClaim<Fr, KzgOutput>;
type ClearWitness<'a> = HomomorphicBatchWitness<'a, Fr, KzgOpeningHint>;

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn kzg_setup(max_num_vars: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let prover = KzgPCS::setup(
        &mut rng,
        1usize << max_num_vars,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier = KzgPCS::verifier_setup(&prover);
    (prover, verifier)
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

fn clear_claims<'a>(
    polynomials: &'a [Polynomial<Fr>],
    point: &Point<HIGH_TO_LOW, Fr>,
    setup: &<KzgPCS as CommitmentScheme>::ProverSetup,
) -> (Vec<ClearClaim>, ClearWitness<'a>) {
    let mut claims = Vec::with_capacity(polynomials.len());
    let mut witness = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        let (commitment, hint) = <KzgPCS as CommitmentScheme>::commit(polynomial, setup);
        claims.push(VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), polynomial.evaluate(point)),
        });
        witness.push((polynomial as &dyn MultilinearPoly<Fr>, hint));
    }
    (claims, witness)
}

#[test]
fn hyperkzg_homomorphic_batch_roundtrip_clear_many_polynomials() {
    let (polynomials, point) = homomorphic_polynomials(5, 4, 0x70_55_1e);
    let (prover_setup, verifier_setup) = kzg_setup(point.len());
    let (claims, witness) = clear_claims(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-batch");
    let proof = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        witness,
        &mut prover_transcript,
    )
    .expect("HyperKZG homomorphic batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-batch");
    <HomomorphicKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        claims,
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
    let (claims, witness) = clear_claims(&polynomials, &point, &prover_setup);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-batch-tamper");
    let proof = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims.clone(),
        witness,
        &mut prover_transcript,
    )
    .expect("HyperKZG homomorphic batch proof should be produced");

    let mut tampered = claims;
    tampered[0].evaluation.value += fr(1);

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-batch-tamper");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        tampered,
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered HyperKZG batch value should fail");
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_mismatched_point() {
    let (polynomials, point) = homomorphic_polynomials(4, 3, 0x72_00_01);
    let (prover_setup, _) = kzg_setup(point.len());
    let (mut claims, witness) = clear_claims(&polynomials, &point, &prover_setup);
    claims[2].evaluation.point = Point::new(vec![fr(2), fr(3), fr(5)]);

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-batch-point-mismatch");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        witness,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_witness_count_mismatch() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x72_00_02);
    let (prover_setup, _) = kzg_setup(point.len());
    let (claims, mut witness) = clear_claims(&polynomials, &point, &prover_setup);
    let _dropped = witness.pop();

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-batch-witness-count");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        witness,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_homomorphic_batch_rejects_wrong_witness_dimension() {
    let (polynomials, point) = homomorphic_polynomials(3, 3, 0x72_00_03);
    let (prover_setup, _) = kzg_setup(point.len());
    let (claims, mut witness) = clear_claims(&polynomials, &point, &prover_setup);
    let wrong_witness = Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)]);
    witness[1].0 = &wrong_witness;

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-batch-witness-dim");
    let result = <HomomorphicKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        claims,
        witness,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
