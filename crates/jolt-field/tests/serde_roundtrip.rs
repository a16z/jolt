//! Wire-format guarantees for the Solinas types: bincode round-trips, exact
//! per-element sizes (`NUM_BYTES`, no per-element overhead), and rejection of
//! non-canonical encodings.
#![cfg(feature = "solinas")]
#![expect(clippy::unwrap_used)]

use jolt_field::{
    CanonicalRepr, Ext2, FieldCore, FpExt4, FpExt8, Prime128Offset275, Prime32Offset99,
    Prime64Offset59,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::de::DeserializeOwned;
use serde::Serialize;

type F32 = Prime32Offset99;
type F64 = Prime64Offset59;
type F128 = Prime128Offset275;

fn assert_roundtrip_with_size<T>(value: &T, expected_len: usize)
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let bytes = bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap();
    assert_eq!(
        bytes.len(),
        expected_len,
        "serialized size must be exactly the canonical byte length"
    );
    let (decoded, consumed): (T, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(consumed, bytes.len());
    assert_eq!(&decoded, value);
}

#[test]
fn prime_field_elements_encode_to_num_bytes() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..32 {
        assert_roundtrip_with_size(&F32::random(&mut rng), <F32 as CanonicalRepr>::NUM_BYTES);
        assert_roundtrip_with_size(&F64::random(&mut rng), <F64 as CanonicalRepr>::NUM_BYTES);
        assert_roundtrip_with_size(&F128::random(&mut rng), <F128 as CanonicalRepr>::NUM_BYTES);
    }
}

#[test]
fn extension_field_elements_encode_to_num_coeffs_times_num_bytes() {
    let mut rng = StdRng::seed_from_u64(8);
    for _ in 0..16 {
        assert_roundtrip_with_size(&Ext2::<F64>::random(&mut rng), 2 * 8);
        assert_roundtrip_with_size(&FpExt4::<F32>::random(&mut rng), 4 * 4);
        assert_roundtrip_with_size(&FpExt8::<F32>::random(&mut rng), 8 * 4);
    }
}

#[test]
fn vectors_add_only_a_single_length_prefix() {
    let mut rng = StdRng::seed_from_u64(9);
    for n in [0usize, 1, 17, 200] {
        let v: Vec<F64> = (0..n).map(|_| F64::random(&mut rng)).collect();
        let bytes = bincode::serde::encode_to_vec(&v, bincode::config::standard()).unwrap();
        // bincode's standard config uses a varint length prefix: 1 byte for
        // lengths below 251.
        let prefix = if n < 251 { 1 } else { 3 };
        assert_eq!(bytes.len(), prefix + n * <F64 as CanonicalRepr>::NUM_BYTES);
        let (decoded, _): (Vec<F64>, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(decoded, v);
    }
}

#[test]
fn non_canonical_encodings_are_rejected() {
    // The modulus itself is not a canonical representative.
    let p32: u32 = u32::MAX - 98;
    let bytes =
        bincode::serde::encode_to_vec(p32.to_le_bytes(), bincode::config::standard()).unwrap();
    assert!(
        bincode::serde::decode_from_slice::<F32, _>(&bytes, bincode::config::standard()).is_err()
    );

    let p64: u64 = u64::MAX - 58;
    let bytes =
        bincode::serde::encode_to_vec(p64.to_le_bytes(), bincode::config::standard()).unwrap();
    assert!(
        bincode::serde::decode_from_slice::<F64, _>(&bytes, bincode::config::standard()).is_err()
    );

    let p128: u128 = u128::MAX - 274;
    let bytes =
        bincode::serde::encode_to_vec(p128.to_le_bytes(), bincode::config::standard()).unwrap();
    assert!(
        bincode::serde::decode_from_slice::<F128, _>(&bytes, bincode::config::standard()).is_err()
    );
}

#[test]
fn canonical_transcript_bytes_and_serde_bytes_agree_for_prime_fields() {
    // For the prime fields both encodings are the canonical little-endian
    // representative; pin that so neither drifts.
    let mut rng = StdRng::seed_from_u64(10);
    for _ in 0..16 {
        let x = F128::random(&mut rng);
        let wire = bincode::serde::encode_to_vec(x, bincode::config::standard()).unwrap();
        assert_eq!(wire, x.to_bytes_le_vec());
    }
}

#[test]
fn from_u64_sanity() {
    let x = F32::from_u64(42);
    let bytes = bincode::serde::encode_to_vec(x, bincode::config::standard()).unwrap();
    assert_eq!(bytes, 42u32.to_le_bytes());
}
