//! Serialization round-trip tests for all BN254 types.

use jolt_crypto::{Bn254, Bn254G1, Bn254G2, Bn254GT, JoltGroup, PairingGroup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
fn g1_json_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let g = Bn254::random_g1(&mut rng);
    let json = serde_json::to_string(&g).expect("serialize");
    let recovered: Bn254G1 = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(g, recovered);
}

#[test]
fn g1_bincode_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let g = Bn254::random_g1(&mut rng);
    let bytes = bincode::serialize(&g).expect("serialize");
    let recovered: Bn254G1 = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(g, recovered);
}

#[test]
fn g1_zero_roundtrip() {
    let z = Bn254G1::zero();
    let bytes = bincode::serialize(&z).expect("serialize");
    let recovered: Bn254G1 = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(z, recovered);
    assert!(recovered.is_zero());
}

#[test]
fn g2_json_roundtrip() {
    let g = Bn254::g2_generator();
    let json = serde_json::to_string(&g).expect("serialize");
    let recovered: Bn254G2 = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(g, recovered);
}

#[test]
fn g2_bincode_roundtrip() {
    let g = Bn254::g2_generator().scalar_mul(&Fr::from_u64(42));
    let bytes = bincode::serialize(&g).expect("serialize");
    let recovered: Bn254G2 = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(g, recovered);
}

#[test]
fn gt_json_roundtrip() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let gt = Bn254::pairing(&g1, &g2);

    let json = serde_json::to_string(&gt).expect("serialize");
    let recovered: Bn254GT = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(gt, recovered);
}

#[test]
fn gt_bincode_roundtrip() {
    let g1 = Bn254::g1_generator();
    let g2 = Bn254::g2_generator();
    let gt = Bn254::pairing(&g1, &g2);

    let bytes = bincode::serialize(&gt).expect("serialize");
    let recovered: Bn254GT = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(gt, recovered);
}

#[test]
fn g1_generator_roundtrip() {
    let g = Bn254::g1_generator();
    let bytes = bincode::serialize(&g).expect("serialize");
    let recovered: Bn254G1 = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(g, recovered);
}

#[test]
fn multiple_g1_elements_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let elements: Vec<Bn254G1> = (0..10).map(|_| Bn254::random_g1(&mut rng)).collect();

    let bytes = bincode::serialize(&elements).expect("serialize");
    let recovered: Vec<Bn254G1> = bincode::deserialize(&bytes).expect("deserialize");
    assert_eq!(elements, recovered);
}
