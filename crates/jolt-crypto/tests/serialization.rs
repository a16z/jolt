#![expect(clippy::expect_used)]

use ark_serialize::CanonicalDeserialize;
use jolt_crypto::{Bn254, Bn254G1, Bn254G2, Bn254GT, JoltGroup, PedersenSetup};
use num_traits::Zero;
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
fn group_identity_with_nonzero_x_is_rejected() {
    let config = bincode::config::standard();

    let mut g1 = bincode::serde::encode_to_vec(Bn254G1::identity(), config).expect("serialize G1");
    assert_eq!(g1[0], 32, "bincode byte length");
    g1[1] = 1;
    let decoded = ark_bn254::G1Projective::deserialize_compressed(&g1[1..]).expect("arkworks G1");
    assert!(decoded.is_zero(), "arkworks discards x for infinity");
    assert!(bincode::serde::decode_from_slice::<Bn254G1, _>(&g1, config).is_err());

    let mut g2 = bincode::serde::encode_to_vec(Bn254G2::identity(), config).expect("serialize G2");
    assert_eq!(g2[0], 64, "bincode byte length");
    g2[1] = 1;
    let decoded = ark_bn254::G2Projective::deserialize_compressed(&g2[1..]).expect("arkworks G2");
    assert!(decoded.is_zero(), "arkworks discards x for infinity");
    assert!(bincode::serde::decode_from_slice::<Bn254G2, _>(&g2, config).is_err());
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
