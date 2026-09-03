#![expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index fixtures and fail loudly on rejected valid proofs"
)]

use jolt_crypto::{Bn254, JoltGroup};
use jolt_field::{Field, Fr, Ring};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_zeromorph::{
    ZeromorphCommitment, ZeromorphProof, ZeromorphProverSetup, ZeromorphScheme,
    ZeromorphVerifierSetup,
};
use num_traits::One;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Scheme = ZeromorphScheme<Bn254>;

fn setup(num_vars: usize) -> (ZeromorphProverSetup<Bn254>, ZeromorphVerifierSetup<Bn254>) {
    Scheme::setup_from_secret(
        Fr::from_u64(7),
        num_vars,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    )
    .unwrap()
}

fn random_instance(num_vars: usize, seed: u64) -> (Polynomial<Fr>, Vec<Fr>, Fr) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let polynomial = Polynomial::random(num_vars, &mut rng);
    let point = (0..num_vars).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
    let evaluation = polynomial.evaluate(&point);
    (polynomial, point, evaluation)
}

fn prove(
    setup: &ZeromorphProverSetup<Bn254>,
    polynomial: &Polynomial<Fr>,
    point: &[Fr],
    evaluation: Fr,
) -> ZeromorphProof<Bn254> {
    let mut transcript = Blake2bTranscript::new(b"zeromorph-test");
    <Scheme as CommitmentScheme>::open(
        polynomial,
        point,
        evaluation,
        setup,
        None,
        &mut transcript,
    )
    .unwrap()
}

fn verify(
    setup: &ZeromorphVerifierSetup<Bn254>,
    commitment: &ZeromorphCommitment<Bn254>,
    point: &[Fr],
    evaluation: Fr,
    proof: &ZeromorphProof<Bn254>,
) -> Result<(), OpeningsError> {
    let mut transcript = Blake2bTranscript::new(b"zeromorph-test");
    <Scheme as CommitmentScheme>::verify(
        commitment,
        point,
        evaluation,
        proof,
        setup,
        &mut transcript,
    )
}

#[test]
fn round_trip_at_required_arities() {
    for num_vars in [4, 10, 20] {
        let (pk, vk) = setup(num_vars);
        let (polynomial, point, evaluation) = random_instance(num_vars, num_vars as u64);
        let (commitment, ()) =
            <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
        let proof = prove(&pk, &polynomial, &point, evaluation);
        verify(&vk, &commitment, &point, evaluation, &proof).unwrap();
    }
}

#[test]
fn wrong_claims_and_tampering_reject() {
    let num_vars = 6;
    let (pk, vk) = setup(num_vars);
    let (polynomial, point, evaluation) = random_instance(num_vars, 11);
    let (commitment, ()) = <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
    let proof = prove(&pk, &polynomial, &point, evaluation);

    assert!(verify(
        &vk,
        &commitment,
        &point,
        evaluation + Fr::one(),
        &proof
    )
    .is_err());
    let mut wrong_point = point.clone();
    wrong_point[2] += Fr::one();
    assert!(verify(&vk, &commitment, &wrong_point, evaluation, &proof).is_err());

    let mut tampered = proof.clone();
    tampered.quotient_commitments[3] += Bn254::g1_generator();
    assert!(verify(&vk, &commitment, &point, evaluation, &tampered).is_err());
    let mut tampered = proof;
    tampered.lifted_degree_quotients[0] += Bn254::g1_generator();
    assert!(verify(&vk, &commitment, &point, evaluation, &tampered).is_err());
}

#[test]
fn commitment_order_matches_hyperkzg() {
    let beta = Fr::from_u64(7);
    let evaluations = [
        Fr::from_u64(2),
        Fr::from_u64(3),
        Fr::from_u64(5),
        Fr::from_u64(11),
    ];
    let (pk, _) = Scheme::setup_from_secret(
        beta,
        2,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    )
    .unwrap();
    let polynomial = Polynomial::new(evaluations.to_vec());
    let (zeromorph, ()) = <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
    let (hyperkzg, ()) = <HyperKZGScheme<Bn254> as CommitmentScheme>::commit(
        &polynomial,
        pk.kzg_setup(),
    )
    .unwrap();
    assert_eq!(zeromorph, hyperkzg);

    let expected_scalar = evaluations
        .iter()
        .rev()
        .fold(Fr::from_u64(0), |value, coefficient| {
            value * beta + *coefficient
        });
    assert_eq!(
        zeromorph.point(),
        Bn254::g1_generator().scalar_mul(&expected_scalar)
    );
}

#[test]
fn compressed_payload_size_is_ell_plus_two_g1() {
    let (pk, _) = setup(4);
    let (polynomial, point, evaluation) = random_instance(4, 17);
    let proof = prove(&pk, &polynomial, &point, evaluation);
    let encoded_g1 = postcard::to_stdvec(&proof.opening_proof).unwrap();
    assert_eq!(encoded_g1[0], 32, "postcard byte-string length prefix");
    assert_eq!(encoded_g1.len() - 1, 32, "compressed BN254 G1 payload");
    assert_eq!(proof.compressed_payload_bytes(32), (4 + 2) * 32);
}

#[test]
fn multi_point_round_trip_and_tamper() {
    let num_vars = 8;
    let (pk, vk) = setup(num_vars);
    let (polynomial, _, _) = random_instance(num_vars, 23);
    let mut rng = ChaCha20Rng::seed_from_u64(29);
    let points = (0..3)
        .map(|_| (0..num_vars).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let evaluations = points
        .iter()
        .map(|point| polynomial.evaluate(point))
        .collect::<Vec<_>>();
    let commitment = Scheme::commit(&pk, polynomial.evaluations()).unwrap();

    let mut prover_transcript = Blake2bTranscript::new(b"zeromorph-multi");
    let proof = Scheme::open_multi(
        &pk,
        polynomial.evaluations(),
        &points,
        &evaluations,
        &mut prover_transcript,
    )
    .unwrap();
    let mut verifier_transcript = Blake2bTranscript::new(b"zeromorph-multi");
    Scheme::verify_multi(
        &vk,
        &commitment,
        &points,
        &evaluations,
        &proof,
        &mut verifier_transcript,
    )
    .unwrap();
    assert_eq!(
        proof.compressed_payload_bytes(32),
        (3 * (num_vars + 1) + 1) * 32
    );

    let mut tampered = proof;
    tampered.quotient_commitments[num_vars + 4] += Bn254::g1_generator();
    let mut verifier_transcript = Blake2bTranscript::new(b"zeromorph-multi");
    assert!(Scheme::verify_multi(
        &vk,
        &commitment,
        &points,
        &evaluations,
        &tampered,
        &mut verifier_transcript,
    )
    .is_err());
}
