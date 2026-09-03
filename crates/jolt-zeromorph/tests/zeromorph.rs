#![expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "tests index fixtures and fail loudly on rejected valid proofs"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr, Ring};
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Label, Transcript};
use jolt_zeromorph::{
    ZeromorphCommitment, ZeromorphError, ZeromorphProof, ZeromorphProverSetup, ZeromorphScheme,
    ZeromorphVerifierSetup,
};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type Scheme = ZeromorphScheme<Bn254>;

#[derive(Default)]
struct FixedTranscript {
    next_challenge: usize,
}

impl Transcript for FixedTranscript {
    type Challenge = Fr;

    fn new(_label: &'static [u8]) -> Self {
        Self::default()
    }

    fn append_bytes(&mut self, _bytes: &[u8]) {}

    fn challenge(&mut self) -> Self::Challenge {
        let challenges = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ];
        let challenge = challenges[self.next_challenge];
        self.next_challenge += 1;
        challenge
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

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

fn eval_univariate(coefficients: &[Fr], point: Fr) -> Fr {
    coefficients
        .iter()
        .rev()
        .fold(Fr::zero(), |value, coefficient| value * point + coefficient)
}

fn phi_at(point: Fr, num_vars: usize) -> Fr {
    let mut result = Fr::one();
    let mut power = point;
    for _ in 0..num_vars {
        result *= Fr::one() + power;
        power *= power;
    }
    result
}

fn divide_by_linear(polynomial: &[Fr], root: Fr) -> Vec<Fr> {
    let mut quotient = vec![Fr::zero(); polynomial.len() - 1];
    let mut carry = polynomial[polynomial.len() - 1];
    quotient[polynomial.len() - 2] = carry;
    for index in (1..polynomial.len() - 1).rev() {
        carry = polynomial[index] + root * carry;
        quotient[index - 1] = carry;
    }
    quotient
}

fn prove(
    setup: &ZeromorphProverSetup<Bn254>,
    polynomial: &Polynomial<Fr>,
    commitment: ZeromorphCommitment<Bn254>,
    point: &[Fr],
    evaluation: Fr,
) -> ZeromorphProof<Bn254> {
    let mut transcript = Blake2bTranscript::new(b"zeromorph-test");
    <Scheme as CommitmentScheme>::open(
        polynomial,
        point,
        evaluation,
        setup,
        Some(commitment),
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
        assert_eq!(pk.g1_powers().len(), 1 << num_vars);
        let (polynomial, point, evaluation) = random_instance(num_vars, num_vars as u64);
        let (commitment, hint) =
            <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
        let proof = prove(&pk, &polynomial, hint, &point, evaluation);
        verify(&vk, &commitment, &point, evaluation, &proof).unwrap();
    }
}

#[test]
fn wrong_claims_reject() {
    let num_vars = 6;
    let (pk, vk) = setup(num_vars);
    let (polynomial, point, evaluation) = random_instance(num_vars, 11);
    let (commitment, hint) = <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
    let proof = prove(&pk, &polynomial, hint, &point, evaluation);

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
}

#[test]
fn fixed_challenges_isolate_pairing_tampering() {
    let num_vars = 4;
    let (pk, vk) = setup(num_vars);
    let (polynomial, point, evaluation) = random_instance(num_vars, 13);
    let commitment = Scheme::commit(&pk, polynomial.evaluations()).unwrap();
    let mut prover_transcript = FixedTranscript::new(b"fixed");
    let proof = Scheme::open(
        &pk,
        polynomial.evaluations(),
        &commitment,
        &point,
        evaluation,
        &mut prover_transcript,
    )
    .unwrap();

    for tampered in [
        {
            let mut tampered = proof.clone();
            tampered.quotient_commitments[3] += Bn254::g1_generator();
            tampered
        },
        {
            let mut tampered = proof.clone();
            tampered.lifted_degree_quotients[0] += Bn254::g1_generator();
            tampered
        },
        {
            let mut tampered = proof.clone();
            tampered.opening_proof += Bn254::g1_generator();
            tampered
        },
    ] {
        let mut verifier_transcript = FixedTranscript::new(b"fixed");
        assert!(matches!(
            Scheme::verify(
                &vk,
                &commitment,
                &point,
                evaluation,
                &tampered,
                &mut verifier_transcript,
            ),
            Err(ZeromorphError::PairingCheckFailed)
        ));
    }
}

#[test]
fn statement_binding_rejects_adaptive_evaluation_forgery() {
    let num_vars = 3;
    let (pk, vk) = setup(num_vars);
    let polynomial = Polynomial::new(
        (1..=1 << num_vars)
            .map(Fr::from_u64)
            .collect::<Vec<_>>(),
    );
    let commitment = Scheme::commit(&pk, polynomial.evaluations()).unwrap();
    let identity = Bn254G1::identity();

    let mut omitted_statement = Blake2bTranscript::new(b"adaptive-forgery");
    omitted_statement.append(&Label(b"zeromorph-quotients"));
    for _ in 0..num_vars {
        omitted_statement.append(&identity);
    }
    omitted_statement.append(&Label(b"zeromorph-y"));
    let _y: Fr = omitted_statement.challenge();
    omitted_statement.append(&Label(b"zeromorph-lifted"));
    omitted_statement.append(&identity);
    omitted_statement.append(&Label(b"zeromorph-x"));
    let x: Fr = omitted_statement.challenge();
    omitted_statement.append(&Label(b"zeromorph-z"));
    let z: Fr = omitted_statement.challenge();
    omitted_statement.append(&Label(b"zeromorph-rho"));
    let _rho: Fr = omitted_statement.challenge();

    let phi = phi_at(x, num_vars);
    let claimed_eval = eval_univariate(polynomial.evaluations(), x) * phi.inverse().unwrap();
    let point = vec![Fr::zero(); num_vars];
    assert_ne!(claimed_eval, polynomial.evaluate(&point));
    let mut identity_polynomial = polynomial
        .evaluations()
        .iter()
        .map(|coefficient| z * coefficient)
        .collect::<Vec<_>>();
    identity_polynomial[0] -= z * claimed_eval * phi;
    assert_eq!(eval_univariate(&identity_polynomial, x), Fr::zero());
    let witness = divide_by_linear(&identity_polynomial, x);
    let opening_proof = Bn254::g1_affine_msm(&pk.g1_powers()[1..], &witness);
    let forged_proof = ZeromorphProof {
        quotient_commitments: vec![identity; num_vars],
        lifted_degree_quotients: vec![identity],
        opening_proof,
    };

    let mut verifier_transcript = Blake2bTranscript::new(b"adaptive-forgery");
    assert!(matches!(
        Scheme::verify(
            &vk,
            &commitment,
            &point,
            claimed_eval,
            &forged_proof,
            &mut verifier_transcript,
        ),
        Err(ZeromorphError::PairingCheckFailed)
    ));
}

#[test]
fn arity_one_zero_and_constant_tables() {
    let (pk, vk) = setup(1);
    assert_eq!(pk.g1_powers().len(), 2);
    for value in [Fr::zero(), Fr::from_u64(9)] {
        let polynomial = Polynomial::new(vec![value, value]);
        let point = [Fr::from_u64(13)];
        let (commitment, hint) = <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
        let proof = prove(&pk, &polynomial, hint, &point, value);
        verify(&vk, &commitment, &point, value, &proof).unwrap();
        assert!(proof.quotient_commitments[0].is_identity());
        assert!(proof.lifted_degree_quotients[0].is_identity());
    }
}

#[test]
fn boolean_point_order_is_high_to_low() {
    let (pk, vk) = setup(2);
    let polynomial = Polynomial::new(
        [10, 20, 30, 40]
            .map(Fr::from_u64)
            .to_vec(),
    );
    let point = [Fr::one(), Fr::zero()];
    let (commitment, hint) = <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
    let proof = prove(&pk, &polynomial, hint, &point, Fr::from_u64(30));
    verify(&vk, &commitment, &point, Fr::from_u64(30), &proof).unwrap();
    assert!(verify(&vk, &commitment, &point, Fr::from_u64(20), &proof).is_err());
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
    let (zeromorph, _) = <Scheme as CommitmentScheme>::commit(&polynomial, &pk).unwrap();
    let hyperkzg_setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        beta,
        evaluations.len(),
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let (hyperkzg, ()) = <HyperKZGScheme<Bn254> as CommitmentScheme>::commit(
        &polynomial,
        &hyperkzg_setup,
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
    let commitment = Scheme::commit(&pk, polynomial.evaluations()).unwrap();
    let proof = prove(&pk, &polynomial, commitment, &point, evaluation);
    let encoded_g1 = postcard::to_stdvec(&proof.opening_proof).unwrap();
    assert_eq!(encoded_g1[0], 32, "postcard byte-string length prefix");
    assert_eq!(encoded_g1.len() - 1, 32, "compressed BN254 G1 payload");
    assert_eq!(proof.compressed_payload_bytes(32), (4 + 2) * 32);
    let identity = Bn254G1::identity();
    let three_point_proof = ZeromorphProof::<Bn254> {
        quotient_commitments: vec![identity; 3 * 20],
        lifted_degree_quotients: vec![identity; 3],
        opening_proof: identity,
    };
    assert_eq!(three_point_proof.compressed_payload_bytes(32), 2_048);
}

#[test]
fn multi_point_round_trip() {
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
        &commitment,
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

}
