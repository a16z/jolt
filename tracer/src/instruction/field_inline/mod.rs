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
use jolt_platform::FieldInlineModulus;
use jolt_program::field_inline::FieldEncodedValue;

// Execute in the proof field: fp128 for Akita, BN254 Fr for Dory.
#[cfg(not(feature = "fp128-field-inline"))]
type ProofField = Fr;
#[cfg(feature = "fp128-field-inline")]
type ProofField = Prime128OffsetA7F7;

/// The field field-inline instructions execute over. jolt-host forwards it to
/// field-inline guest builds, whose SDK conversions prove they were compiled
/// for this field.
#[cfg(not(feature = "fp128-field-inline"))]
pub const FIELD_INLINE_MODULUS: FieldInlineModulus = FieldInlineModulus::Bn254;
#[cfg(feature = "fp128-field-inline")]
pub const FIELD_INLINE_MODULUS: FieldInlineModulus = FieldInlineModulus::Fp128;

// The guest-visible limb table is a copy; pin it to the executing field in
// every field-inline build.
const _: () = assert!(
    limbs_eq(FIELD_INLINE_MODULUS.limbs(), &ProofField::MODULUS_LIMBS),
    "FIELD_INLINE_MODULUS limbs must equal the proof field modulus"
);

const fn limbs_eq(left: &[u64], right: &[u64]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

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
    // zero.
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

    /// A guest compiled for another field fails its conversions' modulus
    /// binding, because that field's modulus is a nonzero element here; and the
    /// pinned limbs are exactly this field's modulus, whose predecessor is the
    /// canonical encoding of -1.
    #[test]
    fn only_the_executing_modulus_reduces_to_zero() {
        let encode_limbs = |limbs: &[u64]| {
            let mut encoded = FieldEncodedValue::zero();
            for (chunk, limb) in encoded.bytes_le.chunks_exact_mut(8).zip(limbs) {
                chunk.copy_from_slice(&limb.to_le_bytes());
            }
            encoded
        };
        for modulus in [FieldInlineModulus::Bn254, FieldInlineModulus::Fp128] {
            assert_eq!(
                decode_field::<ProofField>(encode_limbs(modulus.limbs()))
                    == ProofField::from_u64(0),
                modulus == FIELD_INLINE_MODULUS,
                "{modulus:?}"
            );
        }

        let mut below = ProofField::MODULUS_LIMBS;
        below[0] -= 1;
        assert_eq!(
            ProofField::from_bytes_le_checked(
                &encode_limbs(&below).bytes_le[..ProofField::NUM_BYTES]
            ),
            Some(ProofField::from_u64(0) - ProofField::from_u64(1))
        );
        assert!(FIELD_INLINE_MODULUS.is_canonical(&below));
        assert!(!FIELD_INLINE_MODULUS.is_canonical(&ProofField::MODULUS_LIMBS));
    }
}
