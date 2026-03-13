//! Pedersen commitment scheme tests over BN254 G1.

use jolt_crypto::{Bn254, Bn254G1, JoltCommitment, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn deterministic_setup(count: usize) -> PedersenSetup<Bn254G1> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead);
    let message_generators: Vec<Bn254G1> = (0..count).map(|_| Bn254::random_g1(&mut rng)).collect();
    let blinding_generator = Bn254::random_g1(&mut rng);
    PedersenSetup::new(message_generators, blinding_generator)
}

#[test]
fn commit_verify_roundtrip() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(1);

    let values: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
    assert!(Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &values,
        &blinding
    ));
}

#[test]
fn wrong_values_rejected() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(2);

    let values: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);

    let mut wrong_values = values.clone();
    wrong_values[0] += Fr::from_u64(1);
    assert!(!Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &wrong_values,
        &blinding
    ));
}

#[test]
fn wrong_blinding_rejected() {
    let setup = deterministic_setup(4);
    let mut rng = ChaCha20Rng::seed_from_u64(3);

    let values: Vec<Fr> = (0..4).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);

    let wrong_blinding = blinding + Fr::from_u64(1);
    assert!(!Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &values,
        &wrong_blinding
    ));
}

#[test]
fn different_blinding_different_commitment() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(4);

    let values: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let r1 = Fr::random(&mut rng);
    let r2 = Fr::random(&mut rng);

    let c1 = Pedersen::<Bn254G1>::commit(&setup, &values, &r1);
    let c2 = Pedersen::<Bn254G1>::commit(&setup, &values, &r2);
    assert_ne!(
        c1, c2,
        "different blindings should produce different commitments"
    );
}

#[test]
fn commitment_is_binding() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(5);

    let v1: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let v2: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let c1 = Pedersen::<Bn254G1>::commit(&setup, &v1, &blinding);
    let c2 = Pedersen::<Bn254G1>::commit(&setup, &v2, &blinding);
    assert_ne!(
        c1, c2,
        "different messages with same blinding should differ"
    );
}

#[test]
fn additive_homomorphism() {
    // C(m₁, r₁) + C(m₂, r₂) == C(m₁+m₂, r₁+r₂)
    let setup = deterministic_setup(3);
    let mut rng = ChaCha20Rng::seed_from_u64(6);

    let v1: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let v2: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let r1 = Fr::random(&mut rng);
    let r2 = Fr::random(&mut rng);

    let c1 = Pedersen::<Bn254G1>::commit(&setup, &v1, &r1);
    let c2 = Pedersen::<Bn254G1>::commit(&setup, &v2, &r2);

    let v_sum: Vec<Fr> = v1.iter().zip(v2.iter()).map(|(a, b)| *a + *b).collect();
    let r_sum = r1 + r2;
    let c_sum = Pedersen::<Bn254G1>::commit(&setup, &v_sum, &r_sum);

    assert_eq!(c1 + c2, c_sum, "Pedersen should be additively homomorphic");
}

#[test]
fn zero_blinding_commit() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(7);

    let values: Vec<Fr> = (0..2).map(|_| Fr::random(&mut rng)).collect();
    let zero = Fr::from_u64(0);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &zero);
    // Without blinding, commitment is purely MSM of message generators.
    let expected = Bn254G1::msm(&setup.message_generators[..2], &values);
    assert_eq!(commitment, expected);
}

#[test]
fn capacity_returns_generator_count() {
    let setup = deterministic_setup(10);
    assert_eq!(Pedersen::<Bn254G1>::capacity(&setup), 10);
}

#[test]
#[should_panic(expected = "exceeds generator count")]
fn commit_panics_on_exceeding_capacity() {
    let setup = deterministic_setup(2);
    let mut rng = ChaCha20Rng::seed_from_u64(9);

    let values: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let _ = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
}

#[test]
#[should_panic(expected = "at least one message generator")]
fn setup_panics_on_empty_generators() {
    let _ = PedersenSetup::new(Vec::<Bn254G1>::new(), Bn254G1::identity());
}

#[test]
fn partial_values_uses_prefix_generators() {
    let setup = deterministic_setup(8);
    let mut rng = ChaCha20Rng::seed_from_u64(8);

    let values: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
    let blinding = Fr::random(&mut rng);

    let commitment = Pedersen::<Bn254G1>::commit(&setup, &values, &blinding);
    assert!(Pedersen::<Bn254G1>::verify(
        &setup,
        &commitment,
        &values,
        &blinding
    ));
}
