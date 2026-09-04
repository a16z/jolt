#![expect(
    non_camel_case_types,
    reason = "Tracer concrete instruction names mirror generated Jolt instruction constants"
)]

#[cfg(not(feature = "fp128-field-inline"))]
use jolt_field::Fr;
#[cfg(feature = "fp128-field-inline")]
use jolt_field::Prime128OffsetA7F7;
use jolt_field::{CanonicalEncoding, Field};
use jolt_program::field_inline::{
    FieldEncodedValue, FieldInlineBridge, FieldInlineTraceData, FieldRegisterRead,
    FieldRegisterWrite,
};
use jolt_riscv::{FieldInlineOp, SourceInstructionKind};
use serde::{Deserialize, Serialize};

use super::{
    format::{format_field_inline::FormatFieldInline, InstructionFormat},
    RAMAccess, RISCVInstruction, RISCVTrace,
};
use crate::emulator::cpu::Cpu;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineCycleData {
    pub trace: Option<FieldInlineTraceData>,
}

impl From<FieldInlineCycleData> for RAMAccess {
    fn from(_value: FieldInlineCycleData) -> Self {
        Self::NoOp
    }
}

macro_rules! field_instruction {
    ($name:ident, $op:expr, $source_kind:expr) => {
        #[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
        pub struct $name {
            pub address: u64,
            pub operands: FormatFieldInline,
            pub virtual_sequence_remaining: Option<u16>,
            pub is_first_in_sequence: bool,
            pub is_compressed: bool,
        }

        impl RISCVInstruction for $name {
            const MASK: u32 = $op.instruction_mask();
            const MATCH: u32 = $op.instruction_match();

            type Format = FormatFieldInline;
            type RAMAccess = FieldInlineCycleData;

            fn operands(&self) -> &Self::Format {
                &self.operands
            }

            fn source_kind(&self) -> SourceInstructionKind {
                $source_kind
            }

            fn new(word: u32, address: u64, _validate: bool, is_compressed: bool) -> Self {
                Self {
                    address,
                    operands: FormatFieldInline::parse(word),
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed,
                }
            }

            fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess) {
                ram_access.trace = Some(execute_field_inline($op, self.operands, cpu));
            }
        }

        impl RISCVTrace for $name {}

        impl From<super::SourceInstructionRow> for $name {
            fn from(row: super::SourceInstructionRow) -> Self {
                let mut operands = FormatFieldInline::from(row.operands);
                operands.op = Some($op);
                Self {
                    address: row.address as u64,
                    operands,
                    virtual_sequence_remaining: None,
                    is_first_in_sequence: false,
                    is_compressed: row.is_compressed,
                }
            }
        }
    };
}

field_instruction!(
    FIELD_ADD,
    FieldInlineOp::Add,
    SourceInstructionKind::FIELD_ADD
);
field_instruction!(
    FIELD_SUB,
    FieldInlineOp::Sub,
    SourceInstructionKind::FIELD_SUB
);
field_instruction!(
    FIELD_MUL,
    FieldInlineOp::Mul,
    SourceInstructionKind::FIELD_MUL
);
field_instruction!(
    FIELD_INV,
    FieldInlineOp::Inv,
    SourceInstructionKind::FIELD_INV
);
field_instruction!(
    FIELD_ASSERT_EQ,
    FieldInlineOp::AssertEq,
    SourceInstructionKind::FIELD_ASSERT_EQ
);
field_instruction!(
    FIELD_LOAD_FROM_X,
    FieldInlineOp::LoadFromX,
    SourceInstructionKind::FIELD_LOAD_FROM_X
);
field_instruction!(
    FIELD_STORE_TO_X,
    FieldInlineOp::StoreToX,
    SourceInstructionKind::FIELD_STORE_TO_X
);
field_instruction!(
    FIELD_LOAD_IMM,
    FieldInlineOp::LoadImm,
    SourceInstructionKind::FIELD_LOAD_IMM
);

// The proof field the tracer executes over — the single selection point:
// `fp128-field-inline` (the akita chain) selects the fp128 akita field, the
// homomorphic (Dory) builds keep BN254 Fr. FieldValueEncoding::ACTIVE in
// jolt-program flips with the same feature, so a trace and its metadata
// always agree on the encoding. Everything below is generic over F; no other
// line names a concrete field.
#[cfg(not(feature = "fp128-field-inline"))]
type ProofField = Fr;
#[cfg(feature = "fp128-field-inline")]
type ProofField = Prime128OffsetA7F7;

fn execute_field_inline(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    execute_over::<ProofField>(op, operands, cpu)
}

fn execute_over<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    match op {
        FieldInlineOp::Add => execute_binary::<F>(op, operands, cpu, |left, right| left + right),
        FieldInlineOp::Sub => execute_binary::<F>(op, operands, cpu, |left, right| left - right),
        FieldInlineOp::Mul => {
            let mut trace = execute_binary::<F>(op, operands, cpu, |left, right| left * right);
            trace.product = trace.rd.map(|write| write.post_value);
            trace
        }
        FieldInlineOp::Inv => execute_inverse::<F>(op, operands, cpu),
        FieldInlineOp::AssertEq => execute_assert_eq::<F>(op, operands, cpu),
        FieldInlineOp::LoadFromX => execute_load_from_x::<F>(op, operands, cpu),
        FieldInlineOp::StoreToX => execute_store_to_x::<F>(op, operands, cpu),
        FieldInlineOp::LoadImm => execute_load_imm(op, operands, cpu),
    }
}

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

fn execute_inverse<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let pre_value = cpu.field_registers.read(rd_register);
    // inv(0) fails closed: the R1CS row `IsFieldInv * (FieldInvProduct - 1) = 0`
    // demands `rs1 * rd == 1`, which is unsatisfiable for rs1 = 0, so a trace
    // containing FIELD_INV(0) can never be proven. Trapping here surfaces the
    // guest bug at trace time instead of as a downstream sumcheck failure; the
    // guest-side API is responsible for guarding zero before emitting FIELD_INV.
    let inverse = decode_field::<F>(rs1_value).inverse().unwrap_or_else(|| {
        panic!(
            "FIELD_INV of zero at pc 0x{:x} (fr{}): the inverse constraint is \
             unsatisfiable for a zero operand; guard the guest-side inverse",
            cpu.read_pc(),
            rs1_register,
        )
    });
    let post_value = encode_field(inverse);
    cpu.field_registers.write(rd_register, post_value);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: rs1_register,
            value: rs1_value,
        }),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value,
        }),
        inv_product: Some(encode_field(decode_field::<F>(rs1_value) * inverse)),
        ..Default::default()
    }
}

fn execute_assert_eq<F: CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rs1_register = operands.rs1.unwrap_or(0);
    let rs2_register = operands.rs2.unwrap_or(0);
    let rs1_value = cpu.field_registers.read(rs1_register);
    let rs2_value = cpu.field_registers.read(rs2_register);
    assert_eq!(
        decode_field::<F>(rs1_value),
        decode_field::<F>(rs2_value),
        "field-inline assert_eq failed"
    );
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
        ..Default::default()
    }
}

fn execute_load_from_x<F: Field + CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let x_register = operands.rs1.unwrap_or(0);
    let rd_register = operands.rd.unwrap_or(0);
    let x_value = cpu.read_register(x_register) as u64;
    let field_value = encode_field(F::from_u64(x_value));
    let pre_value = cpu.field_registers.read(rd_register);
    cpu.field_registers.write(rd_register, field_value);
    FieldInlineTraceData {
        op: Some(op),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value: field_value,
        }),
        bridge: Some(FieldInlineBridge::LoadFromX {
            x_register,
            x_value,
            field_value,
        }),
        ..Default::default()
    }
}

fn execute_store_to_x<F: CanonicalEncoding>(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let field_register = operands.rs1.unwrap_or(0);
    let x_register = operands.rd.unwrap_or(0);
    // x0 discards the write, which the bridge row cannot express (it equates the
    // x-register write with the field value); preprocessing rejects the same
    // encoding, so trap at trace time like the other guest faults.
    assert!(
        x_register != 0,
        "FIELD_STORE_TO_X to x0 at pc 0x{:x}: x0 discards the write, store to a real register",
        cpu.read_pc(),
    );
    let field_value = cpu.field_registers.read(field_register);
    // store-to-x is a range-bound bridge: the instruction carries the advice
    // lookup flags, so the constraint system pins the x-register write to
    // `RangeCheck(FieldRs1Value)`, satisfiable only when the field value already
    // fits in 64 bits (`jolt-r1cs` `field_constraints` module doc). A wider
    // value has no proof, so trap here at trace time. Full-width extraction is
    // the advice pattern's job (advice limbs + Horner + FIELD_ASSERT_EQ).
    let x_value = decode_field::<F>(field_value)
        .to_u64_checked()
        .unwrap_or_else(|| {
            panic!(
                "FIELD_STORE_TO_X of a value wider than 64 bits at pc 0x{:x} (fr{}): \
                 the store bridge only supports field values < 2^64; extract wide \
                 values through the advice pattern instead",
                cpu.read_pc(),
                field_register,
            )
        });
    cpu.write_register(x_register as usize, x_value as i64);
    FieldInlineTraceData {
        op: Some(op),
        rs1: Some(FieldRegisterRead {
            register: field_register,
            value: field_value,
        }),
        bridge: Some(FieldInlineBridge::StoreToX {
            field_register,
            field_value,
            x_register,
            x_value,
        }),
        ..Default::default()
    }
}

fn execute_load_imm(
    op: FieldInlineOp,
    operands: FormatFieldInline,
    cpu: &mut Cpu,
) -> FieldInlineTraceData {
    let rd_register = operands.rd.unwrap_or(0);
    // Decoded FIELD_LOAD_IMM immediates are zero-extended 12-bit values (0..=4095);
    // anything else can only arrive through synthetic instruction construction and
    // indicates a caller bug, so fail loudly instead of loading zero.
    let value = u64::try_from(operands.imm).map_or_else(
        |_| {
            panic!(
                "FIELD_LOAD_IMM with out-of-range immediate {} at pc 0x{:x}: decoded \
                 field-inline immediates are zero-extended 12-bit values",
                operands.imm,
                cpu.read_pc(),
            )
        },
        FieldEncodedValue::from_u64,
    );
    let pre_value = cpu.field_registers.read(rd_register);
    cpu.field_registers.write(rd_register, value);
    FieldInlineTraceData {
        op: Some(op),
        rd: Some(FieldRegisterWrite {
            register: rd_register,
            pre_value,
            post_value: value,
        }),
        ..Default::default()
    }
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
