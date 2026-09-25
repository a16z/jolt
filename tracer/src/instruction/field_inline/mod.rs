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
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineTraceData, FieldRegisterRead, FieldRegisterWrite,
};
use jolt_riscv::FieldInlineOp;
use serde::{Deserialize, Serialize};

use super::{format::format_field_inline::FormatFieldInline, RAMAccess, RAMRead};
use crate::emulator::cpu::Cpu;

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FieldInlineCycleData {
    pub trace: Option<FieldInlineTraceData>,
    /// The word read by a memory-sourced load; `None` for every other op.
    pub ram_read: Option<RAMRead>,
}

impl From<FieldInlineCycleData> for RAMAccess {
    fn from(value: FieldInlineCycleData) -> Self {
        value.ram_read.map_or(Self::NoOp, Self::Read)
    }
}

impl From<FieldInlineTraceData> for FieldInlineCycleData {
    fn from(trace: FieldInlineTraceData) -> Self {
        Self {
            trace: Some(trace),
            ram_read: None,
        }
    }
}

// The tracer and FieldValueEncoding::ACTIVE select the same proof field:
// fp128 for Akita builds, BN254 Fr for Dory builds.
#[cfg(not(feature = "fp128-field-inline"))]
type ProofField = Fr;
#[cfg(feature = "fp128-field-inline")]
type ProofField = Prime128OffsetA7F7;

fn execute_binary<F: CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
    f: impl FnOnce(F, F) -> F,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rs2_register = operands.rs2.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let rs2_value = cpu.field_registers.read(rs2_register);
    let pre_value = cpu.field_registers.read(rd_register);
    let post_value = encode_field(f(decode_field(rs1_value), decode_field(rs2_value)));
    cpu.field_registers.write(rd_register, post_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rs2: Some(FieldRegisterRead {
            register: rs2_register,
            value: rs2_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value,
        }),
        ..Default::default()
    }
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
