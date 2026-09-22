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
use jolt_poly::CompressedPoly;
use jolt_r1cs::ConstraintMatrices;
use jolt_spartan_prover::prove;
use jolt_spartan_verifier::{SpartanError, SpartanKey};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};

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
fn transcript() -> Blake2bTranscript {
    Blake2bTranscript::new(b"spartan-test")
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
        max_public_degree: 3,
    })
    .unwrap()
}

fn check_backend<PCS: CommitmentScheme<Field = Fr>>(pk: &PCS::ProverSetup, vk: &PCS::VerifierSetup)
where
    PCS::Output: AppendToTranscript,
{
    let key = key();
    let mut pt = transcript();
    let proof = prove::<PCS>(&key, &public(), &witness(), pk, &mut pt).unwrap();
    let mut vt = transcript();
    key.verify::<PCS>(&public(), &proof, vk, &mut vt).unwrap();
    assert_eq!(pt.challenge(), vt.challenge());
    let bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
    let (decoded, used) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(used, bytes.len());
    key.verify::<PCS>(&public(), &decoded, vk, &mut transcript())
        .unwrap();
    assert!(key
        .verify::<PCS>(&[Fr::from_u64(33)], &proof, vk, &mut transcript())
        .is_err());
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
    let proof =
        prove::<HyperKZGScheme>(&key, &[], &[Fr::from_u64(3)], &pk, &mut transcript()).unwrap();
    key.verify::<HyperKZGScheme>(&[], &proof, &vk, &mut transcript())
        .unwrap();
}

#[test]
fn bad_inputs_and_malformed_matrices_reject() {
    let (pk, _) = hyperkzg_setup();
    assert!(matches!(
        prove::<HyperKZGScheme>(&key(), &[], &witness(), &pk, &mut transcript()),
        Err(SpartanError::PublicInputs)
    ));
    assert!(matches!(
        prove::<HyperKZGScheme>(&key(), &public(), &[], &pk, &mut transcript()),
        Err(SpartanError::WitnessLength)
    ));
    assert!(matches!(
        prove::<HyperKZGScheme>(&key(), &public(), &[Fr::zero(); 3], &pk, &mut transcript()),
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
fn matrix_encoding_and_policy_are_bound_even_for_equivalent_relations() {
    let (pk, vk) = hyperkzg_setup();
    let proof =
        prove::<HyperKZGScheme>(&key(), &public(), &witness(), &pk, &mut transcript()).unwrap();
    let mut m = matrices();
    m.a[0][0].1 += Fr::one();
    m.c[0][0].1 += Fr::one();
    let equivalent = SpartanKey::new(m, 1, [19; 32]).unwrap();
    assert!(equivalent
        .verify::<HyperKZGScheme>(&public(), &proof, &vk, &mut transcript())
        .is_err());
    let another_policy = SpartanKey::new(matrices(), 1, [20; 32]).unwrap();
    assert!(another_policy
        .verify::<HyperKZGScheme>(&public(), &proof, &vk, &mut transcript())
        .is_err());
}

#[test]
fn proof_claims_commitment_and_opening_are_bound() {
    let key = key();
    let (pk, vk) = hyperkzg_setup();
    let proof =
        prove::<HyperKZGScheme>(&key, &public(), &witness(), &pk, &mut transcript()).unwrap();
    let reject = |proof: &_| {
        assert!(key
            .verify::<HyperKZGScheme>(&public(), proof, &vk, &mut transcript())
            .is_err());
    };
    for index in 0..3 {
        let mut changed = proof.clone();
        changed.outer_evaluations[index] += Fr::one();
        reject(&changed);
    }
    let mut changed = proof.clone();
    changed.witness_evaluation += Fr::one();
    reject(&changed);
    let mut changed = proof.clone();
    changed.witness_commitment += Bn254::g1_generator();
    reject(&changed);
    let mut changed = proof.clone();
    changed.opening.w[0] += Bn254::g1_generator();
    reject(&changed);
    for outer in [true, false] {
        let mut changed = proof.clone();
        let round = if outer {
            &mut changed.outer.round_polynomials[0]
        } else {
            &mut changed.inner.round_polynomials[0]
        };
        let mut coefficients = round.coeffs_except_linear_term().to_vec();
        coefficients[0] += Fr::one();
        *round = CompressedPoly::new(coefficients);
        reject(&changed);
    }
}

#[test]
fn exact_round_counts_and_degrees_are_enforced() {
    let key = key();
    let (pk, vk) = hyperkzg_setup();
    let proof =
        prove::<HyperKZGScheme>(&key, &public(), &witness(), &pk, &mut transcript()).unwrap();
    for outer in [true, false] {
        let mut changed = proof.clone();
        let rounds = if outer {
            &mut changed.outer.round_polynomials
        } else {
            &mut changed.inner.round_polynomials
        };
        rounds[0] = CompressedPoly::new(vec![Fr::zero(); if outer { 4 } else { 3 }]);
        assert!(matches!(
            key.verify::<HyperKZGScheme>(&public(), &changed, &vk, &mut transcript()),
            Err(SpartanError::Sumcheck(
                SumcheckError::DegreeBoundExceeded { .. }
            ))
        ));
        let mut changed = proof.clone();
        let rounds = if outer {
            &mut changed.outer.round_polynomials
        } else {
            &mut changed.inner.round_polynomials
        };
        let _ = rounds.pop();
        assert!(matches!(
            key.verify::<HyperKZGScheme>(&public(), &changed, &vk, &mut transcript()),
            Err(SpartanError::Sumcheck(
                SumcheckError::WrongNumberOfRounds { .. }
            ))
        ));
    }
}
