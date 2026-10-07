#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures fail loudly"
)]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup};
use jolt_field::{CanonicalBytes, CanonicalDecode, Fr, One, Ring, Zero};
use jolt_hyperkzg::{
    HyperKZGError, HyperKZGProverSetup, HyperKZGScheme, HyperKZGSetupParams, HyperKZGVerifierSetup,
};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::Polynomial;
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, TranscriptError, VerifierTranscript,
};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-hyperkzg/test");
const SESSION: &[u8] = b"hyperkzg-test";

fn setup_params(beta: u64, capacity: usize) -> HyperKZGSetupParams {
    let beta = Fr::from_u64(beta);
    HyperKZGSetupParams {
        setup_id: [42; 32],
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

fn prover() -> ProverTranscript<Blake2b512> {
    ProverTranscript::new(&PROTOCOL, SESSION)
}

fn verifier(narg: &[u8]) -> VerifierTranscript<'_, Blake2b512> {
    VerifierTranscript::new(&PROTOCOL, SESSION, narg)
}

fn open(
    poly: &Polynomial<Fr>,
    point: &[Fr],
    evaluation: Fr,
    pk: &HyperKZGProverSetup,
    hint: Option<Bn254G1>,
) -> Result<Vec<u8>, OpeningsError> {
    let mut transcript = prover();
    HyperKZGScheme::open(poly, point, evaluation, pk, hint, &mut transcript)?;
    Ok(transcript.finish())
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

/// The opening's prover messages, decoded from its argument string: binary
/// fold commitments, evaluations at `[r, -r, r^2]`, and KZG witnesses.
#[derive(Clone)]
struct Messages {
    com: Vec<Bn254G1>,
    v: [Vec<Fr>; 3],
    w: [Bn254G1; 3],
}

impl Messages {
    fn decode(narg: &[u8], num_vars: usize) -> Self {
        let mut rest = narg;
        let mut take = |len: usize| {
            let (head, tail) = rest.split_at(len);
            rest = tail;
            head
        };
        let mut point = || Bn254G1::from_bytes_le_checked(take(Bn254G1::NUM_BYTES)).unwrap();
        let com = (1..num_vars).map(|_| point()).collect();
        let mut scalar = || Fr::from_bytes_le_checked(take(Fr::NUM_BYTES)).unwrap();
        let v = std::array::from_fn(|_| (0..num_vars).map(|_| scalar()).collect());
        let w = std::array::from_fn(|_| {
            Bn254G1::from_bytes_le_checked(take(Bn254G1::NUM_BYTES)).unwrap()
        });
        assert!(rest.is_empty());
        Self { com, v, w }
    }

    fn encode(&self) -> Vec<u8> {
        let mut narg = Vec::new();
        for point in &self.com {
            narg.extend(point.to_bytes_le_vec());
        }
        for value in self.v.iter().flatten() {
            narg.extend(value.to_bytes_le_vec());
        }
        for point in &self.w {
            narg.extend(point.to_bytes_le_vec());
        }
        narg
    }
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
        let mut pt = prover();
        HyperKZGScheme::open(&poly, &point, evaluation, &pk, Some(hint), &mut pt).unwrap();
        let narg = pt.narg().to_vec();
        assert_eq!(
            narg.len(),
            (num_vars - 1 + 3) * Bn254G1::NUM_BYTES + 3 * num_vars * Fr::NUM_BYTES
        );
        let mut vt = verifier(&narg);
        HyperKZGScheme::verify(&commitment, &point, evaluation, &vk, &mut vt).unwrap();
        assert_eq!(pt.challenge_bytes::<32>(), vt.challenge_bytes::<32>());
        vt.finish().unwrap();
    }
}

#[test]
fn coefficient_commitment_has_known_answer() {
    let (pk, _) = setup(7, 4);
    let poly = Polynomial::new([1, 2, 3, 4].map(Fr::from_u64).to_vec());
    let (commitment, _) = HyperKZGScheme::commit(&poly, &pk).unwrap();
    assert_eq!(
        commitment,
        Bn254::g1_generator().scalar_mul(&Fr::from_u64(1534))
    );
}

struct Fixture {
    commitment: Bn254G1,
    point: Vec<Fr>,
    evaluation: Fr,
    messages: Messages,
    vk: HyperKZGVerifierSetup,
}

impl Fixture {
    fn new() -> Self {
        let (pk, vk) = setup(7, 16);
        let table = (1..=16).map(Fr::from_u64).collect::<Vec<_>>();
        let point = [2, 3, 5, 7].map(Fr::from_u64).to_vec();
        let evaluation = multilinear_oracle(&table, &point);
        let poly = Polynomial::new(table);
        let (commitment, _) = HyperKZGScheme::commit(&poly, &pk).unwrap();
        let narg = open(&poly, &point, evaluation, &pk, None).unwrap();
        Self {
            commitment,
            messages: Messages::decode(&narg, point.len()),
            point,
            evaluation,
            vk,
        }
    }

    /// Verifies `narg` as the whole argument string of one opening.
    fn verify(&self, narg: &[u8]) -> Result<(), HyperKZGError> {
        let mut transcript = verifier(narg);
        HyperKZGScheme::verify_opening(
            &self.commitment,
            &self.point,
            self.evaluation,
            &self.vk,
            &mut transcript,
        )?;
        Ok(transcript.finish()?)
    }

    fn verify_messages(&self, messages: &Messages) -> Result<(), HyperKZGError> {
        self.verify(&messages.encode())
    }
}

#[test]
fn every_transmitted_proof_component_is_bound() {
    let f = Fixture::new();
    f.verify_messages(&f.messages).unwrap();
    for row in 0..3 {
        for column in 0..f.point.len() {
            let mut messages = f.messages.clone();
            messages.v[row][column] += Fr::one();
            assert!(f.verify_messages(&messages).is_err());
        }
        let mut messages = f.messages.clone();
        messages.w[row] += Bn254::g1_generator();
        assert!(f.verify_messages(&messages).is_err());
    }
    for index in 0..f.messages.com.len() {
        let mut messages = f.messages.clone();
        messages.com[index] += Bn254::g1_generator();
        assert!(f.verify_messages(&messages).is_err());
    }
}

#[test]
fn statement_changes_reject() {
    let mut f = Fixture::new();
    f.evaluation += Fr::one();
    assert!(f.verify_messages(&f.messages).is_err());
    f.evaluation -= Fr::one();
    for index in 0..f.point.len() {
        f.point[index] += Fr::one();
        assert!(f.verify_messages(&f.messages).is_err());
        f.point[index] -= Fr::one();
    }
    f.commitment += Bn254::g1_generator();
    assert!(f.verify_messages(&f.messages).is_err());
    f.commitment -= Bn254::g1_generator();
    f.vk = setup(11, 16).1;
    assert!(f.verify_messages(&f.messages).is_err());
    f.vk = setup(7, 32).1;
    assert!(f.verify_messages(&f.messages).is_err());
    let mut params = setup_params(7, 16);
    params.setup_id = [43; 32];
    f.vk = HyperKZGScheme::setup(params).unwrap().1;
    assert!(f.verify_messages(&f.messages).is_err());
}

#[test]
fn absent_hint_is_recomputed_and_present_hint_is_bound() {
    let f = Fixture::new();
    f.verify_messages(&f.messages).unwrap();
    let (pk, _) = setup(7, 16);
    let poly = Polynomial::new((1..=16).map(Fr::from_u64).collect::<Vec<_>>());
    let hint = f.commitment + Bn254::g1_generator();
    let narg = open(&poly, &f.point, f.evaluation, &pk, Some(hint)).unwrap();
    assert!(f.verify(&narg).is_err());
}

#[test]
fn malformed_shape_and_arity_return_errors_before_transcript_mutation() {
    let f = Fixture::new();
    let narg = f.messages.encode();
    assert_eq!(
        f.verify(&narg[..narg.len() - 1]),
        Err(HyperKZGError::Transcript(TranscriptError::Truncated))
    );
    let trailing = [narg.as_slice(), &[0]].concat();
    assert_eq!(
        f.verify(&trailing),
        Err(HyperKZGError::Transcript(TranscriptError::TrailingBytes))
    );
    for point in [
        vec![],
        vec![Fr::one(); 5],
        vec![Fr::one(); usize::BITS as usize],
    ] {
        let mut actual = verifier(&narg);
        assert_eq!(
            HyperKZGScheme::verify_opening(&f.commitment, &point, f.evaluation, &f.vk, &mut actual),
            Err(HyperKZGError::InvalidArity)
        );
        assert_eq!(actual.remaining(), narg.len());
        assert_eq!(
            actual.challenge_bytes::<32>(),
            verifier(&narg).challenge_bytes::<32>()
        );
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
    let (pk, _) = setup(7, 4);
    let poly = Polynomial::new(vec![Fr::one(); 4]);
    assert!(open(&poly, &[Fr::zero(); 2], Fr::zero(), &pk, None).is_err());
    assert!(open(&poly, &[Fr::zero()], Fr::one(), &pk, None).is_err());
    let zero = Polynomial::new(vec![Fr::zero(); 4]);
    let (commitment, _) = HyperKZGScheme::commit(&zero, &pk).unwrap();
    let narg = open(&zero, &[Fr::one(); 2], Fr::zero(), &pk, None).unwrap();
    let mut transcript = verifier(&narg);
    HyperKZGScheme::verify(
        &commitment,
        &[Fr::one(); 2],
        Fr::zero(),
        &HyperKZGScheme::verifier_setup(&pk),
        &mut transcript,
    )
    .unwrap();
    transcript.finish().unwrap();
}
