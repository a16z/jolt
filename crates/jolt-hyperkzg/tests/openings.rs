#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures fail loudly"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    HyperKZGError, HyperKZGProof, HyperKZGProverSetup, HyperKZGScheme, HyperKZGSetupParams,
    HyperKZGVerifierSetup,
};
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

fn setup_params(beta: u64, capacity: usize) -> HyperKZGSetupParams {
    let beta = Fr::from_u64(beta);
    HyperKZGSetupParams {
        setup_id: [42; 32],
        max_public_degree: capacity.saturating_sub(1) as u64,
        g1_powers: std::iter::successors(Some(Fr::one()), |power| Some(*power * beta))
            .take(capacity)
            .map(|power| Bn254::g1_generator().scalar_mul(&power))
            .collect(),
        g2: Bn254::g2_generator(),
        beta_g2: Bn254::g2_generator().scalar_mul(&beta),
    }
}

fn setup(beta: u64, capacity: usize) -> (HyperKZGProverSetup, HyperKZGVerifierSetup) {
    HyperKZGScheme::setup(setup_params(beta, capacity)).unwrap()
}

fn transcript() -> Blake2bTranscript {
    Blake2bTranscript::new(b"hyperkzg-test")
}

fn multilinear_oracle(table: &[Fr], point: &[Fr]) -> Fr {
    table
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let weight = point
                .iter()
                .enumerate()
                .map(|(bit, coordinate)| {
                    if index & (1 << (point.len() - 1 - bit)) == 0 {
                        Fr::one() - coordinate
                    } else {
                        *coordinate
                    }
                })
                .product::<Fr>();
            *value * weight
        })
        .sum()
}

#[test]
fn independent_evaluations_and_transcripts_match() {
    let (pk, vk) = setup(7, 64);
    for num_vars in 1..=6 {
        let table = (0..1usize << num_vars)
            .map(|i| Fr::from_u64((i * i + 3) as u64))
            .collect::<Vec<_>>();
        let point = (0..num_vars)
            .map(|i| Fr::from_u64(i as u64 + 2))
            .collect::<Vec<_>>();
        let evaluation = multilinear_oracle(&table, &point);
        let poly = Polynomial::new(table);
        let (commitment, hint) = HyperKZGScheme::commit(&poly, &pk).unwrap();
        let mut pt = transcript();
        let proof =
            HyperKZGScheme::open(&poly, &point, evaluation, &pk, Some(hint), &mut pt).unwrap();
        let bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        let (proof, consumed): (HyperKZGProof, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        let mut vt = transcript();
        HyperKZGScheme::verify(&commitment, &point, evaluation, &proof, &vk, &mut vt).unwrap();
        assert_eq!(pt.challenge(), vt.challenge());
    }
}

#[test]
fn coefficient_commitment_has_known_answer() {
    let (pk, _) = setup(7, 4);
    let poly = Polynomial::new([1, 2, 3, 4].map(Fr::from_u64).to_vec());
    let (commitment, ()) = HyperKZGScheme::commit(&poly, &pk).unwrap();
    assert_eq!(
        commitment,
        Bn254::g1_generator().scalar_mul(&Fr::from_u64(1534))
    );
}

struct Fixture {
    commitment: Bn254G1,
    point: Vec<Fr>,
    evaluation: Fr,
    proof: HyperKZGProof,
    vk: HyperKZGVerifierSetup,
}

impl Fixture {
    fn new() -> Self {
        let (pk, vk) = setup(7, 16);
        let table = (1..=16).map(Fr::from_u64).collect::<Vec<_>>();
        let point = [2, 3, 5, 7].map(Fr::from_u64).to_vec();
        let evaluation = multilinear_oracle(&table, &point);
        let poly = Polynomial::new(table);
        let (commitment, ()) = HyperKZGScheme::commit(&poly, &pk).unwrap();
        let proof =
            HyperKZGScheme::open(&poly, &point, evaluation, &pk, None, &mut transcript()).unwrap();
        Self {
            commitment,
            point,
            evaluation,
            proof,
            vk,
        }
    }

    fn verify(&self, proof: &HyperKZGProof) -> Result<(), HyperKZGError> {
        HyperKZGScheme::verify_opening(
            &self.commitment,
            &self.point,
            self.evaluation,
            proof,
            &self.vk,
            &mut transcript(),
        )
    }
}

#[test]
fn every_transmitted_proof_component_is_bound() {
    let f = Fixture::new();
    for row in 0..3 {
        for column in 0..f.point.len() {
            let mut proof = f.proof.clone();
            proof.v[row][column] += Fr::one();
            assert!(f.verify(&proof).is_err());
        }
        let mut proof = f.proof.clone();
        proof.w[row] += Bn254::g1_generator();
        assert!(f.verify(&proof).is_err());
    }
    for index in 0..f.proof.com.len() {
        let mut proof = f.proof.clone();
        proof.com[index] += Bn254::g1_generator();
        assert!(f.verify(&proof).is_err());
    }
}

#[test]
fn statement_changes_reject() {
    let mut f = Fixture::new();
    f.evaluation += Fr::one();
    assert!(f.verify(&f.proof).is_err());
    f.evaluation -= Fr::one();
    for index in 0..f.point.len() {
        f.point[index] += Fr::one();
        assert!(f.verify(&f.proof).is_err());
        f.point[index] -= Fr::one();
    }
    f.commitment += Bn254::g1_generator();
    assert!(f.verify(&f.proof).is_err());
    f.commitment -= Bn254::g1_generator();
    f.vk = setup(11, 16).1;
    assert!(f.verify(&f.proof).is_err());
    f.vk = setup(7, 32).1;
    assert!(f.verify(&f.proof).is_err());
    let mut params = setup_params(7, 16);
    params.setup_id = [43; 32];
    f.vk = HyperKZGScheme::setup(params).unwrap().1;
    assert!(f.verify(&f.proof).is_err());
    let mut params = setup_params(7, 16);
    params.max_public_degree = 31;
    f.vk = HyperKZGScheme::setup(params).unwrap().1;
    assert!(f.verify(&f.proof).is_err());
}

#[test]
fn malformed_shape_and_arity_return_errors_before_transcript_mutation() {
    let f = Fixture::new();
    for row in 0..3 {
        let mut proof = f.proof.clone();
        let _ = proof.v[row].pop();
        assert_eq!(f.verify(&proof), Err(HyperKZGError::ProofShape));
        proof.v[row].extend([Fr::zero(); 2]);
        assert_eq!(f.verify(&proof), Err(HyperKZGError::ProofShape));
    }
    let mut proof = f.proof.clone();
    let _ = proof.com.pop();
    assert_eq!(f.verify(&proof), Err(HyperKZGError::ProofShape));
    for point in [
        vec![],
        vec![Fr::one(); 5],
        vec![Fr::one(); usize::BITS as usize],
    ] {
        let mut actual = transcript();
        assert_eq!(
            HyperKZGScheme::verify_opening(
                &f.commitment,
                &point,
                f.evaluation,
                &f.proof,
                &f.vk,
                &mut actual
            ),
            Err(HyperKZGError::InvalidArity)
        );
        assert_eq!(actual.challenge(), transcript().challenge());
    }
}

#[test]
fn bad_imports_and_false_prover_claims_reject() {
    for capacity in 0..2 {
        assert!(HyperKZGScheme::setup(setup_params(7, capacity)).is_err());
    }
    let mut params = setup_params(7, 4);
    params.g1_powers[2] = Bn254G1::identity();
    assert!(HyperKZGScheme::setup(params).is_err());
    let mut params = setup_params(7, 4);
    params.beta_g2 = JoltGroup::identity();
    assert!(HyperKZGScheme::setup(params).is_err());
    let mut params = setup_params(7, 4);
    params.max_public_degree = 2;
    assert!(HyperKZGScheme::setup(params).is_err());
    let (pk, _) = setup(7, 4);
    let poly = Polynomial::new(vec![Fr::one(); 4]);
    assert!(HyperKZGScheme::open(
        &poly,
        &[Fr::zero(); 2],
        Fr::zero(),
        &pk,
        None,
        &mut transcript()
    )
    .is_err());
    assert!(HyperKZGScheme::open(
        &poly,
        &[Fr::zero()],
        Fr::one(),
        &pk,
        None,
        &mut transcript()
    )
    .is_err());
    let zero = Polynomial::new(vec![Fr::zero(); 4]);
    let (commitment, ()) = HyperKZGScheme::commit(&zero, &pk).unwrap();
    let proof = HyperKZGScheme::open(
        &zero,
        &[Fr::one(); 2],
        Fr::zero(),
        &pk,
        None,
        &mut transcript(),
    )
    .unwrap();
    HyperKZGScheme::verify(
        &commitment,
        &[Fr::one(); 2],
        Fr::zero(),
        &proof,
        &HyperKZGScheme::verifier_setup(&pk),
        &mut transcript(),
    )
    .unwrap();
}

#[derive(Default)]
struct ZeroTranscript;

impl Transcript for ZeroTranscript {
    type Challenge = Fr;

    fn new(_: &'static [u8]) -> Self {
        Self
    }

    fn append_bytes(&mut self, _: &[u8]) {}

    fn challenge(&mut self) -> Fr {
        Fr::zero()
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

#[test]
fn zero_fold_challenge_rejects_on_both_paths() {
    let f = Fixture::new();
    assert_eq!(
        HyperKZGScheme::verify_opening(
            &f.commitment,
            &f.point,
            f.evaluation,
            &f.proof,
            &f.vk,
            &mut ZeroTranscript,
        ),
        Err(HyperKZGError::DegenerateChallenge),
    );
    let (pk, _) = setup(7, 4);
    let poly = Polynomial::new(vec![Fr::one(); 4]);
    assert!(HyperKZGScheme::open(
        &poly,
        &[Fr::one(); 2],
        Fr::one(),
        &pk,
        None,
        &mut ZeroTranscript,
    )
    .is_err());
}
