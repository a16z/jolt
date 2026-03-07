//! Integration tests for Spartan with HyperKZG polynomial commitment scheme.
//!
//! Exercises the full prove → verify pipeline using real BN254 pairing-based
//! commitments instead of MockPCS.

use jolt_crypto::Bn254;
use jolt_field::{Field, Fr};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::CommitmentScheme;
use jolt_spartan::{
    FirstRoundStrategy, SimpleR1CS, SpartanError, SpartanKey, SpartanProver, SpartanVerifier,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

type KzgPCS = HyperKZGScheme<Bn254>;

fn make_setup(max_degree: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let pk = KzgPCS::setup(&mut rng, max_degree, g1, g2);
    let vk = KzgPCS::verifier_setup(&pk);
    (pk, vk)
}

fn prove_and_verify(
    r1cs: &SimpleR1CS<Fr>,
    key: &SpartanKey<Fr>,
    witness: &[Fr],
    pk: &HyperKZGProverSetup<Bn254>,
    vk: &HyperKZGVerifierSetup<Bn254>,
    label: &'static [u8],
) {
    let mut t_p = Blake2bTranscript::new(label);
    let proof = SpartanProver::prove::<KzgPCS, _>(
        r1cs,
        key,
        witness,
        pk,
        &mut t_p,
        FirstRoundStrategy::Standard,
    )
    .expect("proving should succeed");

    let mut t_v = Blake2bTranscript::new(label);
    SpartanVerifier::verify::<KzgPCS, _>(key, &proof, vk, &mut t_v)
        .expect("verification should succeed");
}

// Basic circuits

/// x * x = y with HyperKZG.
#[test]
fn prove_verify_x_squared() {
    let r1cs = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let (pk, vk) = make_setup(key.num_variables_padded);
    let witness = [Fr::from_u64(1), Fr::from_u64(5), Fr::from_u64(25)];
    prove_and_verify(&r1cs, &key, &witness, &pk, &vk, b"kzg-x2");
}

/// Bad witness rejected.
#[test]
fn reject_bad_witness() {
    let r1cs = SimpleR1CS::new(
        1,
        3,
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 1, Fr::from_u64(1))],
        vec![(0, 2, Fr::from_u64(1))],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let (pk, _vk) = make_setup(key.num_variables_padded);

    let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(10)]; // 3*3 != 10
    let mut t = Blake2bTranscript::new(b"kzg-bad");
    let result = SpartanProver::prove::<KzgPCS, _>(
        &r1cs,
        &key,
        &witness,
        &pk,
        &mut t,
        FirstRoundStrategy::Standard,
    );
    assert!(
        matches!(result, Err(SpartanError::ConstraintViolation(0))),
        "expected constraint violation"
    );
}

/// Multiple constraints: x*x=y, y*x=z.
#[test]
fn multiple_constraints() {
    let one = Fr::from_u64(1);
    let r1cs = SimpleR1CS::new(
        2,
        4,
        vec![(0, 1, one), (1, 2, one)],
        vec![(0, 1, one), (1, 1, one)],
        vec![(0, 2, one), (1, 3, one)],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let (pk, vk) = make_setup(key.num_variables_padded);
    // x=2: [1, 2, 4, 8]
    let witness = [one, Fr::from_u64(2), Fr::from_u64(4), Fr::from_u64(8)];
    prove_and_verify(&r1cs, &key, &witness, &pk, &vk, b"kzg-multi");
}

/// Chain multiplication: x^2, x^3, x^4, x^5.
#[test]
fn chain_multiplication() {
    let one = Fr::from_u64(1);
    let r1cs = SimpleR1CS::new(
        4,
        6,
        vec![(0, 1, one), (1, 2, one), (2, 3, one), (3, 4, one)],
        vec![(0, 1, one), (1, 1, one), (2, 1, one), (3, 1, one)],
        vec![(0, 2, one), (1, 3, one), (2, 4, one), (3, 5, one)],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let (pk, vk) = make_setup(key.num_variables_padded);
    // x=3: powers 1, 3, 9, 27, 81, 243
    let witness = [
        one,
        Fr::from_u64(3),
        Fr::from_u64(9),
        Fr::from_u64(27),
        Fr::from_u64(81),
        Fr::from_u64(243),
    ];
    prove_and_verify(&r1cs, &key, &witness, &pk, &vk, b"kzg-chain");
}

// Univariate skip strategy

/// Both strategies produce verifiable proofs.
#[test]
fn uniskip_matches_standard() {
    let one = Fr::from_u64(1);
    let r1cs = SimpleR1CS::new(
        2,
        4,
        vec![(0, 1, one), (1, 2, one)],
        vec![(0, 1, one), (1, 1, one)],
        vec![(0, 2, one), (1, 3, one)],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let (pk, vk) = make_setup(key.num_variables_padded);
    let witness = [one, Fr::from_u64(2), Fr::from_u64(4), Fr::from_u64(8)];

    // Standard
    {
        let mut t_p = Blake2bTranscript::new(b"kzg-std");
        let proof = SpartanProver::prove::<KzgPCS, _>(
            &r1cs, &key, &witness, &pk, &mut t_p, FirstRoundStrategy::Standard,
        )
        .unwrap();
        let mut t_v = Blake2bTranscript::new(b"kzg-std");
        SpartanVerifier::verify::<KzgPCS, _>(&key, &proof, &vk, &mut t_v).unwrap();
    }

    // UnivariateSkip
    {
        let mut t_p = Blake2bTranscript::new(b"kzg-uni");
        let proof = SpartanProver::prove::<KzgPCS, _>(
            &r1cs, &key, &witness, &pk, &mut t_p, FirstRoundStrategy::UnivariateSkip,
        )
        .unwrap();
        let mut t_v = Blake2bTranscript::new(b"kzg-uni");
        SpartanVerifier::verify::<KzgPCS, _>(&key, &proof, &vk, &mut t_v).unwrap();
    }
}

// Relaxed Spartan with HyperKZG

/// Relaxed Spartan with u=1, E=0 (standard instance presented as relaxed).
#[test]
fn relaxed_standard_instance() {
    let one = Fr::from_u64(1);
    // Multi-constraint to avoid degenerate 0-variable error polynomial
    let r1cs = SimpleR1CS::new(
        2,
        4,
        vec![(0, 1, one), (1, 1, one)],
        vec![(0, 1, one), (1, 2, one)],
        vec![(0, 2, one), (1, 3, one)],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let srs_size = std::cmp::max(key.num_variables_padded, key.num_constraints_padded);
    let (pk, vk) = make_setup(srs_size);

    // x=3: [1, 3, 9, 27]
    let witness = [one, Fr::from_u64(3), Fr::from_u64(9), Fr::from_u64(27)];
    let error = vec![Fr::from_u64(0); key.num_constraints_padded];

    let (w_com, ()) = <KzgPCS as CommitmentScheme>::commit(&witness, &pk);
    let (e_com, ()) = <KzgPCS as CommitmentScheme>::commit(&error, &pk);

    let mut t_p = Blake2bTranscript::new(b"kzg-relaxed");
    let proof = SpartanProver::prove_relaxed::<KzgPCS, _>(
        &r1cs, &key, one, &witness, &error, &w_com, &e_com, &pk, &mut t_p,
    )
    .expect("relaxed proving should succeed");

    let mut t_v = Blake2bTranscript::new(b"kzg-relaxed");
    SpartanVerifier::verify_relaxed::<KzgPCS, _>(&key, one, &w_com, &e_com, &proof, &vk, &mut t_v)
        .expect("relaxed verification should succeed");
}

// Randomized property test

/// Random circuits with varying sizes always verify.
#[test]
fn randomized_circuits() {
    for seed in 6000..6005 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let num_constraints = 1 + (seed as usize % 4); // 1..4
        let num_variables = num_constraints + 2; // always enough room
        let one = Fr::from_u64(1);

        // Build a chain of multiplications: z[i+2] = z[1] * z[i+1]
        let mut a_entries = Vec::new();
        let mut b_entries = Vec::new();
        let mut c_entries = Vec::new();
        for i in 0..num_constraints {
            a_entries.push((i, 1, one)); // A picks z[1]
            b_entries.push((i, i + 1, one)); // B picks z[i+1]
            c_entries.push((i, i + 2, one)); // C picks z[i+2]
        }

        let r1cs = SimpleR1CS::new(num_constraints, num_variables, a_entries, b_entries, c_entries);
        let key = SpartanKey::from_r1cs(&r1cs);
        let (pk, vk) = make_setup(key.num_variables_padded);

        // Build valid witness: z[0]=1, z[1]=x (random), z[i+2]=x*z[i+1]
        let mut witness = vec![Fr::from_u64(0); num_variables];
        witness[0] = one;
        witness[1] = Fr::random(&mut rng);
        for i in 0..num_constraints {
            witness[i + 2] = witness[1] * witness[i + 1];
        }

        prove_and_verify(&r1cs, &key, &witness, &pk, &vk, b"kzg-rand");
    }
}
