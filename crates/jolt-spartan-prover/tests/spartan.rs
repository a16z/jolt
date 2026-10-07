#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures fail loudly"
)]

use jolt_crypto::{Bn254, JoltGroup};
use jolt_dory::DoryScheme;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGSetupParams};
use jolt_openings::CommitmentScheme;
use jolt_r1cs::ConstraintMatrices;
use jolt_spartan_prover::prove;
use jolt_spartan_verifier::{SpartanError, SpartanKey};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, TranscriptError, VerifierTranscript,
};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-spartan/test");
const SESSION: &[u8] = b"spartan-test";

fn matrices() -> ConstraintMatrices<Fr> {
    let one = Fr::one();
    ConstraintMatrices::new(
        3,
        5,
        vec![
            vec![(2, one)],
            vec![(3, one)],
            vec![(4, one), (0, Fr::from_u64(5))],
        ],
        vec![vec![(2, one)], vec![(2, one)], vec![(0, one)]],
        vec![vec![(3, one)], vec![(4, one)], vec![(1, one)]],
    )
}

fn key() -> SpartanKey<Fr> {
    SpartanKey::new(matrices(), 1, [19; 32]).unwrap()
}
fn public() -> [Fr; 1] {
    [Fr::from_u64(32)]
}
fn witness() -> [Fr; 3] {
    [3, 9, 27].map(Fr::from_u64)
}

fn hyperkzg_setup() -> (
    <HyperKZGScheme as CommitmentScheme>::ProverSetup,
    <HyperKZGScheme as CommitmentScheme>::VerifierSetup,
) {
    let beta = Fr::from_u64(7);
    HyperKZGScheme::setup(HyperKZGSetupParams {
        g1_powers: std::iter::successors(Some(Fr::one()), |power| Some(*power * beta))
            .take(4)
            .map(|power| Bn254::g1_generator().scalar_mul(&power))
            .collect(),
        g2: Bn254::g2_generator(),
        beta_g2: Bn254::g2_generator().scalar_mul(&beta),
        setup_id: [9; 32],
    })
    .unwrap()
}

fn prove_narg<PCS: CommitmentScheme<Field = Fr>>(
    key: &SpartanKey<Fr>,
    public_inputs: &[Fr],
    witness: &[Fr],
    pk: &PCS::ProverSetup,
) -> Result<Vec<u8>, SpartanError<Fr>> {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    prove::<PCS, _>(key, public_inputs, witness, pk, &mut transcript)?;
    Ok(transcript.finish())
}

/// Verifies `narg` as the whole argument string of one Spartan proof.
fn verify_narg<PCS: CommitmentScheme<Field = Fr>>(
    key: &SpartanKey<Fr>,
    public_inputs: &[Fr],
    vk: &PCS::VerifierSetup,
    narg: &[u8],
) -> Result<(), SpartanError<Fr>> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    key.verify::<PCS, _>(public_inputs, vk, &mut transcript)?;
    Ok(transcript.finish()?)
}

fn check_backend<PCS: CommitmentScheme<Field = Fr>>(
    pk: &PCS::ProverSetup,
    vk: &PCS::VerifierSetup,
) {
    let key = key();
    let mut pt = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    prove::<PCS, _>(&key, &public(), &witness(), pk, &mut pt).unwrap();
    let narg = pt.narg().to_vec();
    let mut vt = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, &narg);
    key.verify::<PCS, _>(&public(), vk, &mut vt).unwrap();
    assert_eq!(pt.challenge_bytes::<32>(), vt.challenge_bytes::<32>());
    vt.finish().unwrap();
    assert!(verify_narg::<PCS>(&key, &[Fr::from_u64(33)], vk, &narg).is_err());
}

#[test]
fn cube_plus_five_with_non_power_of_two_rows_and_witness() {
    let (pk, vk) = hyperkzg_setup();
    check_backend::<HyperKZGScheme>(&pk, &vk);
    let projected = matrices()
        .project_column_range(
            &[1, 2, 3].map(Fr::from_u64),
            2,
            3,
            [2, 3, 5].map(Fr::from_u64),
        )
        .unwrap();
    assert_eq!(projected, [11, 9, 16].map(Fr::from_u64));
}

#[test]
fn same_relation_works_with_dory() {
    let (pk, vk) = DoryScheme::setup(2).unwrap();
    check_backend::<DoryScheme>(&pk, &vk);
}

#[test]
fn one_row_one_witness_and_no_public_input_are_supported() {
    let one = Fr::one();
    let m = ConstraintMatrices::new(
        1,
        2,
        vec![vec![(1, one)]],
        vec![vec![(1, one)]],
        vec![vec![(0, Fr::from_u64(9))]],
    );
    let key = SpartanKey::new(m, 0, [19; 32]).unwrap();
    let (pk, vk) = hyperkzg_setup();
    let narg = prove_narg::<HyperKZGScheme>(&key, &[], &[Fr::from_u64(3)], &pk).unwrap();
    verify_narg::<HyperKZGScheme>(&key, &[], &vk, &narg).unwrap();
}

#[test]
fn bad_inputs_and_malformed_matrices_reject() {
    let (pk, _) = hyperkzg_setup();
    assert!(matches!(
        prove_narg::<HyperKZGScheme>(&key(), &[], &witness(), &pk),
        Err(SpartanError::PublicInputs)
    ));
    assert!(matches!(
        prove_narg::<HyperKZGScheme>(&key(), &public(), &[], &pk),
        Err(SpartanError::WitnessLength)
    ));
    assert!(matches!(
        prove_narg::<HyperKZGScheme>(&key(), &public(), &[Fr::zero(); 3], &pk),
        Err(SpartanError::Unsatisfied(_))
    ));
    let mut m = matrices();
    m.a[0][0].0 = 5;
    assert!(SpartanKey::new(m, 1, [0; 32]).is_err());
    let mut m = matrices();
    let _ = m.b.pop();
    assert!(SpartanKey::new(m, 1, [0; 32]).is_err());
    for public_count in [4, 5, usize::MAX] {
        assert!(SpartanKey::new(matrices(), public_count, [0; 32]).is_err());
    }
    let mut m = matrices();
    m.num_vars = usize::MAX;
    assert!(SpartanKey::new(m, 0, [0; 32]).is_err());
    assert!(SpartanKey::new(
        ConstraintMatrices::<Fr>::new(0, 2, vec![], vec![], vec![]),
        0,
        [0; 32]
    )
    .is_err());
}

#[test]
fn relation_digest_binds_encoding_of_evaluation_equivalent_matrices() {
    let (pk, vk) = hyperkzg_setup();
    let narg = prove_narg::<HyperKZGScheme>(&key(), &public(), &witness(), &pk).unwrap();
    let mut m = matrices();
    m.c[0] = vec![(3, Fr::from_u64(2)), (3, -Fr::one())];
    m.b[0].push((1, Fr::zero()));
    let equivalent = SpartanKey::new(m, 1, [19; 32]).unwrap();
    let equivalent_narg =
        prove_narg::<HyperKZGScheme>(&equivalent, &public(), &witness(), &pk).unwrap();
    verify_narg::<HyperKZGScheme>(&equivalent, &public(), &vk, &equivalent_narg).unwrap();
    assert!(verify_narg::<HyperKZGScheme>(&equivalent, &public(), &vk, &narg).is_err());
    let another_policy = SpartanKey::new(matrices(), 1, [20; 32]).unwrap();
    assert!(verify_narg::<HyperKZGScheme>(&another_policy, &public(), &vk, &narg).is_err());
}

#[test]
fn every_proof_message_is_bound() {
    // Every message in the argument string (the witness commitment, each
    // sumcheck round, the outer and witness evaluations, and the HyperKZG
    // opening) is a 32-byte canonical element, so flipping a bit of any one
    // either breaks its encoding or changes a bound value.
    let key = key();
    let (pk, vk) = hyperkzg_setup();
    let narg = prove_narg::<HyperKZGScheme>(&key, &public(), &witness(), &pk).unwrap();
    verify_narg::<HyperKZGScheme>(&key, &public(), &vk, &narg).unwrap();
    assert_eq!(narg.len() % 32, 0);
    for element in 0..narg.len() / 32 {
        let mut changed = narg.clone();
        changed[element * 32] ^= 1;
        assert!(
            verify_narg::<HyperKZGScheme>(&key, &public(), &vk, &changed).is_err(),
            "flipping element {element} must reject"
        );
    }
}

#[test]
fn truncated_and_extended_arguments_reject() {
    let key = key();
    let (pk, vk) = hyperkzg_setup();
    let narg = prove_narg::<HyperKZGScheme>(&key, &public(), &witness(), &pk).unwrap();
    assert!(verify_narg::<HyperKZGScheme>(&key, &public(), &vk, &narg[..narg.len() - 32]).is_err());
    let extended = [narg.as_slice(), &[0; 32]].concat();
    assert!(matches!(
        verify_narg::<HyperKZGScheme>(&key, &public(), &vk, &extended),
        Err(SpartanError::Transcript(TranscriptError::TrailingBytes))
    ));
}

#[test]
fn unequal_row_and_witness_padding_accepts() {
    let one = Fr::one();
    let m = ConstraintMatrices::new(
        3,
        2,
        vec![vec![(1, one)], vec![(1, one)], vec![(0, one)]],
        vec![vec![(1, one)], vec![(0, one)], vec![(0, one)]],
        vec![
            vec![(0, Fr::from_u64(9))],
            vec![(0, Fr::from_u64(3))],
            vec![(0, one)],
        ],
    );
    let key = SpartanKey::new(m, 0, [19; 32]).unwrap();
    assert_eq!((key.row_vars(), key.witness_vars()), (2, 1));
    let (pk, vk) = hyperkzg_setup();
    let narg = prove_narg::<HyperKZGScheme>(&key, &[], &[Fr::from_u64(3)], &pk).unwrap();
    verify_narg::<HyperKZGScheme>(&key, &[], &vk, &narg).unwrap();
}
