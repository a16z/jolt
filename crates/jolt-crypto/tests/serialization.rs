//! Serialization round-trip tests for all BN254 types.

use jolt_crypto::{Bn254, Bn254G1, Bn254G2, Bn254GT, JoltGroup, PairingGroup, PedersenSetup};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

fn bincode_roundtrip<T: serde::Serialize + serde::de::DeserializeOwned + Eq + std::fmt::Debug>(
    val: &T,
) -> T {
    let config = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(val, config).expect("serialize");
    let (recovered, _): (T, _) =
        bincode::serde::decode_from_slice(&bytes, config).expect("deserialize");
    recovered
}

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
    let recovered = bincode_roundtrip(&g);
    assert_eq!(g, recovered);
}

#[test]
fn g1_identity_roundtrip() {
    let z = Bn254G1::identity();
    let recovered = bincode_roundtrip(&z);
    assert_eq!(z, recovered);
    assert!(recovered.is_identity());
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
    let recovered = bincode_roundtrip(&g);
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
    let recovered = bincode_roundtrip(&gt);
    assert_eq!(gt, recovered);
}

#[test]
fn g1_generator_roundtrip() {
    let g = Bn254::g1_generator();
    let recovered = bincode_roundtrip(&g);
    assert_eq!(g, recovered);
}

#[test]
fn multiple_g1_elements_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let elements: Vec<Bn254G1> = (0..10).map(|_| Bn254::random_g1(&mut rng)).collect();
    let recovered = bincode_roundtrip(&elements);
    assert_eq!(elements, recovered);
}

#[test]
fn g2_identity_roundtrip() {
    let z = Bn254G2::identity();
    let recovered = bincode_roundtrip(&z);
    assert_eq!(z, recovered);
    assert!(recovered.is_identity());
}

#[test]
fn gt_identity_roundtrip() {
    let z = Bn254GT::identity();
    let recovered = bincode_roundtrip(&z);
    assert_eq!(z, recovered);
    assert!(recovered.is_identity());
}

#[test]
fn pedersen_setup_bincode_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let gens: Vec<Bn254G1> = (0..5).map(|_| Bn254::random_g1(&mut rng)).collect();
    let blinding = Bn254::random_g1(&mut rng);
    let setup = PedersenSetup::new(gens, blinding);

    let config = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(&setup, config).expect("serialize");
    let (recovered, _): (PedersenSetup<Bn254G1>, _) =
        bincode::serde::decode_from_slice(&bytes, config).expect("deserialize");
    assert_eq!(setup.message_generators, recovered.message_generators);
    assert_eq!(setup.blinding_generator, recovered.blinding_generator);
}

#[test]
fn pedersen_setup_json_roundtrip() {
    let mut rng = ChaCha20Rng::seed_from_u64(201);
    let gens: Vec<Bn254G1> = (0..3).map(|_| Bn254::random_g1(&mut rng)).collect();
    let blinding = Bn254::random_g1(&mut rng);
    let setup = PedersenSetup::new(gens, blinding);

    let json = serde_json::to_string(&setup).expect("serialize");
    let recovered: PedersenSetup<Bn254G1> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(setup.message_generators, recovered.message_generators);
    assert_eq!(setup.blinding_generator, recovered.blinding_generator);
}
