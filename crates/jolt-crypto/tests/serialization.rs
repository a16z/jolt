#![expect(clippy::expect_used)]

use jolt_crypto::{Bn254, Bn254G1, Bn254GT, JoltGroup, PedersenSetup};
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
fn g1_identity_roundtrip() {
    let z = Bn254G1::identity();
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
