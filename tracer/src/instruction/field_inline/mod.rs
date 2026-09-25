#![expect(
    non_camel_case_types,
    reason = "Tracer concrete instruction names mirror generated Jolt instruction constants"
)]

pub mod add;
pub mod advice_limb;
pub mod assert_eq;
pub mod assert_zero;
pub mod inv;
pub mod load_accumulate_from_memory;
pub mod load_accumulate_from_register;
pub mod load_imm;
pub mod mul;
pub mod sub;

pub use add::FIELD_ADD;
pub use advice_limb::FIELD_ADVICE_LIMB;
pub use assert_eq::FIELD_ASSERT_EQ;
pub use assert_zero::FIELD_ASSERT_ZERO;
pub use inv::FIELD_INV;
pub use load_accumulate_from_memory::FIELD_LOAD_ACCUMULATE_FROM_MEMORY;
pub use load_accumulate_from_register::FIELD_LOAD_ACCUMULATE_FROM_REGISTER;
pub use load_imm::FIELD_LOAD_IMM;
pub use mul::FIELD_MUL;
pub use sub::FIELD_SUB;

#[cfg(not(feature = "fp128-field-inline"))]
use jolt_field::Fr;
#[cfg(feature = "fp128-field-inline")]
use jolt_field::Prime128OffsetA7F7;
use jolt_field::{CanonicalEncoding, Field};
use jolt_program::field_inline::FieldEncodedValue;

// The tracer and FieldValueEncoding::ACTIVE select the same proof field:
// fp128 for Akita builds, BN254 Fr for Dory builds.
#[cfg(not(feature = "fp128-field-inline"))]
type ProofField = Fr;
#[cfg(feature = "fp128-field-inline")]
type ProofField = Prime128OffsetA7F7;

fn accumulate_word<F: Field + CanonicalEncoding>(
    previous: FieldEncodedValue,
    word: u64,
) -> FieldEncodedValue {
    encode_field(decode_field::<F>(previous) * F::from_u128(1u128 << 64) + F::from_u64(word))
}

fn decode_field<F: CanonicalEncoding>(value: FieldEncodedValue) -> F {
    F::from_bytes_le_reduced(&value.bytes_le)
}

fn encode_field<F: CanonicalEncoding>(value: F) -> FieldEncodedValue {
    // A field wider than the fixed 32-byte register buffer cannot ride
    // FieldEncodedValue; reject it at monomorphization, not per encode.
    const { assert!(F::NUM_BYTES <= FieldEncodedValue::BYTE_LEN as usize) }
    let mut encoded = FieldEncodedValue::zero();
    // Narrower fields occupy the low NUM_BYTES; the rest of the buffer stays
    // zero (FieldValueEncoding::byte_len tags the valid width).
    value.to_bytes_le(&mut encoded.bytes_le[..F::NUM_BYTES]);
    encoded
}

#[cfg(test)]
mod tests {
    use jolt_field::{CanonicalBytes, Ring};

    use super::*;

    /// Encode/decode roundtrip over the build's ProofField, including values
    /// above 2^64 (multi-limb) and the buffer-width contract a narrower field
    /// relies on: bytes past NUM_BYTES stay zero, so downstream full-buffer
    /// reduced decodes (jolt-witness) see the same element.
    #[test]
    fn proof_field_roundtrips_through_the_register_encoding() {
        let wide = ProofField::from_u128((1u128 << 64) + 3);
        let values = [
            ProofField::from_u64(0),
            ProofField::from_u64(1),
            ProofField::from_u64(u64::MAX),
            wide,
            ProofField::from_u64(0) - ProofField::from_u64(1),
            wide.inverse().unwrap(),
        ];
        for value in values {
            let encoded = encode_field(value);
            assert!(encoded.bytes_le[ProofField::NUM_BYTES..]
                .iter()
                .all(|byte| *byte == 0));
            assert_eq!(decode_field::<ProofField>(encoded), value);
        }
    }
}
