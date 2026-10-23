#![expect(clippy::unwrap_used, reason = "tests construct canonical fixtures")]
#![cfg(feature = "solinas")]

use jolt_field::solinas::Prime128OffsetA7F7 as Field;
use jolt_field::{CanonicalBytes, CanonicalEncoding};

const MODULUS: u128 = u128::MAX - (0xffff_a7f7 - 1);

#[test]
fn word_canonical_tuple_boundaries_and_consumption() {
    let config = bincode::config::standard();
    for value in [0, 1, (1u128 << 64) - 1, 1u128 << 64, MODULUS - 1] {
        let mut input = value.to_le_bytes().to_vec();
        input.extend_from_slice(&[0xaa, 0xbb]);
        let (field, consumed): (Field, usize) =
            bincode::serde::decode_from_slice(&input, config).unwrap();
        assert_eq!(field.to_u128_checked(), Some(value));
        assert_eq!(consumed, 16);
        assert_eq!(
            bincode::serde::encode_to_vec(field, config).unwrap(),
            value.to_le_bytes()
        );
        let mut big_endian = [0u8; 16];
        field.to_bytes_be(&mut big_endian);
        assert_eq!(big_endian, value.to_be_bytes());
    }
    for value in [MODULUS, MODULUS + 1, u128::MAX] {
        assert!(
            bincode::serde::decode_from_slice::<Field, _>(&value.to_le_bytes(), config).is_err()
        );
    }
}

#[test]
fn word_truncated_and_single_bit_tuples() {
    let config = bincode::config::standard();
    let input = [0u8; 16];
    for length in 0..16 {
        assert!(bincode::serde::decode_from_slice::<Field, _>(&input[..length], config).is_err());
    }
    // Every bit below the modulus is canonical, including the limb boundary.
    for bit in 0..128 {
        let value = 1u128 << bit;
        let (field, consumed): (Field, usize) =
            bincode::serde::decode_from_slice(&value.to_le_bytes(), config).unwrap();
        assert_eq!((field.to_u128_checked(), consumed), (Some(value), 16));
    }
}

#[test]
fn word_scalar_challenge_lengths_and_reduction() {
    assert_eq!(
        Field::from_scalar_challenge_bytes(&[]).to_u128_checked(),
        Some(0)
    );
    assert_eq!(
        Field::from_scalar_challenge_bytes(&[0x34, 0x12]).to_u128_checked(),
        Some(0x1234)
    );
    assert_eq!(
        Field::from_scalar_challenge_bytes(&MODULUS.to_le_bytes()).to_u128_checked(),
        Some(0)
    );
    let mut wide = [0u8; 17];
    wide[16] = 1;
    assert_eq!(
        Field::from_scalar_challenge_bytes(&wide).to_u128_checked(),
        Some(0xffff_a7f7)
    );
}
