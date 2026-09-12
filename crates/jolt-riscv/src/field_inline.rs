//! Structural vocabulary for field-inline Jolt rows.
//!
//! This module names field-inline instructions and their operand roles. It does
//! not execute field arithmetic or define protocol formulas.

#[cfg(feature = "serialization")]
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};

use crate::{JoltInstruction, NormalizedOperands, SourceInstruction};

pub const FIELD_REGISTER_LOG_K: u8 = 4;
pub const FIELD_REGISTER_COUNT: u8 = 1 << FIELD_REGISTER_LOG_K;
pub const FIELD_INLINE_OPCODE: u8 = 0x7b;
pub const FIELD_INLINE_R_TYPE_FUNCT7: u8 = 0;
pub const FIELD_INLINE_LOAD_IMM_FUNCT3: u8 = 7;
/// The memory-sourced loads share `FIELD_LOAD_FROM_X`'s funct3 and are told
/// apart by funct7: bit 6 marks the family, bit 5 selects the high-word
/// (Horner) form, bits 4..0 carry the word offset added to the base register.
pub const FIELD_INLINE_LOAD_WORD_FUNCT3: u8 = 5;
pub const FIELD_INLINE_LOAD_WORD_FUNCT7_FAMILY: u8 = 0x40;
pub const FIELD_INLINE_LOAD_WORD_FUNCT7_HIGH: u8 = 0x20;
pub const FIELD_INLINE_LOAD_WORD_OFFSET_MASK: u8 = 0x1f;
/// Bytes between consecutive word offsets of a memory-sourced load.
pub const FIELD_INLINE_LOAD_WORD_STRIDE: u32 = 8;
/// The limb split shares `FIELD_STORE_TO_X`'s funct3 under funct7 1.
pub const FIELD_INLINE_ADVICE_LIMB_FUNCT3: u8 = 6;
pub const FIELD_INLINE_ADVICE_LIMB_FUNCT7: u8 = 1;

/// The funct7 of a memory-sourced load at `offset_words` (at most 31).
pub const fn field_inline_load_word_funct7(high: bool, offset_words: u8) -> u8 {
    FIELD_INLINE_LOAD_WORD_FUNCT7_FAMILY
        | if high {
            FIELD_INLINE_LOAD_WORD_FUNCT7_HIGH
        } else {
            0
        }
        | (offset_words & FIELD_INLINE_LOAD_WORD_OFFSET_MASK)
}

/// The byte offset a memory-sourced load word adds to its base register.
pub const fn field_inline_load_word_offset(word: u32) -> u32 {
    (((word >> 25) as u8) & FIELD_INLINE_LOAD_WORD_OFFSET_MASK) as u32
        * FIELD_INLINE_LOAD_WORD_STRIDE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum FieldInlineOp {
    Add,
    Sub,
    Mul,
    Inv,
    AssertEq,
    LoadFromX,
    StoreToX,
    LoadImm,
    /// `frd = mem[x_rs1 + offset]`: an `LD` into the scratch x-register `rd`
    /// whose loaded word also lands in field register `rs2`.
    LoadWord,
    /// `frd = frd · 2^64 + mem[x_rs1 + offset]`: the Horner step of a two-limb
    /// operand fused with the load of its low word (same operand roles as
    /// [`Self::LoadWord`]; field register `rs2` is both read and written).
    LoadWordHi,
    /// Supply a 64-bit advice limb `x_rd` with `frs1 = x_rd + 2^64 · frs2`.
    /// The tracer chooses the canonical low limb; the relation permits other
    /// choices. A full readout needs a guest integer check below the modulus.
    AdviceLimb,
}

impl FieldInlineOp {
    pub const fn tag(self) -> u8 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
            Self::Inv => 3,
            Self::AssertEq => 4,
            Self::LoadFromX => 5,
            Self::StoreToX => 6,
            Self::LoadImm => 7,
            Self::LoadWord => 8,
            Self::LoadWordHi => 9,
            Self::AdviceLimb => 10,
        }
    }

    pub const fn funct3(self) -> u8 {
        match self {
            Self::Add => 0,
            Self::Sub => 1,
            Self::Mul => 2,
            Self::Inv => 3,
            Self::AssertEq => 4,
            Self::LoadFromX => 5,
            Self::StoreToX => 6,
            Self::LoadImm => FIELD_INLINE_LOAD_IMM_FUNCT3,
            Self::LoadWord | Self::LoadWordHi => FIELD_INLINE_LOAD_WORD_FUNCT3,
            Self::AdviceLimb => FIELD_INLINE_ADVICE_LIMB_FUNCT3,
        }
    }

    /// The funct7 the encoding matches on; the memory-sourced loads keep their
    /// word offset in the low funct7 bits, which the match ignores.
    pub const fn funct7(self) -> Option<u8> {
        match self {
            Self::LoadImm => None,
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Inv
            | Self::AssertEq
            | Self::LoadFromX
            | Self::StoreToX => Some(FIELD_INLINE_R_TYPE_FUNCT7),
            Self::LoadWord => Some(field_inline_load_word_funct7(false, 0)),
            Self::LoadWordHi => Some(field_inline_load_word_funct7(true, 0)),
            Self::AdviceLimb => Some(FIELD_INLINE_ADVICE_LIMB_FUNCT7),
        }
    }

    pub const fn is_memory_load(self) -> bool {
        matches!(self, Self::LoadWord | Self::LoadWordHi)
    }

    pub const fn instruction_mask(self) -> u32 {
        match self.funct7() {
            None => 0x0000_707f,
            Some(_) if self.is_memory_load() => 0xc000_707f,
            Some(_) => 0xfe00_707f,
        }
    }

    pub const fn instruction_match(self) -> u32 {
        let base = (FIELD_INLINE_OPCODE as u32) | ((self.funct3() as u32) << 12);
        match self.funct7() {
            Some(funct7) => base | ((funct7 as u32) << 25),
            None => base,
        }
    }

    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Add),
            1 => Some(Self::Sub),
            2 => Some(Self::Mul),
            3 => Some(Self::Inv),
            4 => Some(Self::AssertEq),
            5 => Some(Self::LoadFromX),
            6 => Some(Self::StoreToX),
            7 => Some(Self::LoadImm),
            8 => Some(Self::LoadWord),
            9 => Some(Self::LoadWordHi),
            10 => Some(Self::AdviceLimb),
            _ => None,
        }
    }

    pub const fn from_r_type_key(funct7: u8, funct3: u8) -> Option<Self> {
        if funct3 == FIELD_INLINE_LOAD_WORD_FUNCT3
            && funct7 & FIELD_INLINE_LOAD_WORD_FUNCT7_FAMILY != 0
        {
            return if funct7 & FIELD_INLINE_LOAD_WORD_FUNCT7_HIGH != 0 {
                Some(Self::LoadWordHi)
            } else {
                Some(Self::LoadWord)
            };
        }
        match (funct7, funct3) {
            (FIELD_INLINE_R_TYPE_FUNCT7, 0) => Some(Self::Add),
            (FIELD_INLINE_R_TYPE_FUNCT7, 1) => Some(Self::Sub),
            (FIELD_INLINE_R_TYPE_FUNCT7, 2) => Some(Self::Mul),
            (FIELD_INLINE_R_TYPE_FUNCT7, 3) => Some(Self::Inv),
            (FIELD_INLINE_R_TYPE_FUNCT7, 4) => Some(Self::AssertEq),
            (FIELD_INLINE_R_TYPE_FUNCT7, 5) => Some(Self::LoadFromX),
            (FIELD_INLINE_R_TYPE_FUNCT7, 6) => Some(Self::StoreToX),
            (FIELD_INLINE_ADVICE_LIMB_FUNCT7, FIELD_INLINE_ADVICE_LIMB_FUNCT3) => {
                Some(Self::AdviceLimb)
            }
            _ => None,
        }
    }

    pub const fn from_i_type_funct3(funct3: u8) -> Option<Self> {
        match funct3 {
            FIELD_INLINE_LOAD_IMM_FUNCT3 => Some(Self::LoadImm),
            _ => None,
        }
    }

    pub const fn from_word(word: u32) -> Option<Self> {
        let funct3 = ((word >> 12) & 0x7) as u8;
        match Self::from_i_type_funct3(funct3) {
            Some(op) => Some(op),
            None => Self::from_r_type_key(((word >> 25) & 0x7f) as u8, funct3),
        }
    }
}

#[cfg(feature = "serialization")]
impl CanonicalSerialize for FieldInlineOp {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.tag().serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.tag().serialized_size(compress)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for FieldInlineOp {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        Self::from_tag(tag).ok_or(SerializationError::InvalidData)
    }
}

#[cfg(feature = "serialization")]
impl Valid for FieldInlineOp {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum FieldInlineXRegisterRole {
    ReadRs1,
    WriteRd,
    /// A memory-sourced load: `rs1` is the address base and `rd` the scratch
    /// register the loaded word is written to, exactly an `LD`.
    ReadRs1WriteRd,
}

#[cfg(feature = "serialization")]
impl CanonicalSerialize for FieldInlineXRegisterRole {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let tag = match self {
            Self::ReadRs1 => 0u8,
            Self::WriteRd => 1u8,
            Self::ReadRs1WriteRd => 2u8,
        };
        tag.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        0u8.serialized_size(compress)
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for FieldInlineXRegisterRole {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        match u8::deserialize_with_mode(&mut reader, compress, validate)? {
            0 => Ok(Self::ReadRs1),
            1 => Ok(Self::WriteRd),
            2 => Ok(Self::ReadRs1WriteRd),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

#[cfg(feature = "serialization")]
impl Valid for FieldInlineXRegisterRole {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialization",
    derive(CanonicalSerialize, serde::Serialize, serde::Deserialize)
)]
pub struct FieldRegister(pub u8);

impl FieldRegister {
    pub const fn new(index: u8) -> Option<Self> {
        if index < FIELD_REGISTER_COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }
}

#[cfg(feature = "serialization")]
impl Valid for FieldRegister {
    fn check(&self) -> Result<(), SerializationError> {
        // Enforce the `FieldRegister::new` bound on the deserialize path. The derived
        // `Valid` is a no-op for the inner `u8`, which would otherwise admit out-of-range
        // indices from untrusted bytes.
        if self.0 < FIELD_REGISTER_COUNT {
            Ok(())
        } else {
            Err(SerializationError::InvalidData)
        }
    }
}

#[cfg(feature = "serialization")]
impl CanonicalDeserialize for FieldRegister {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let value = Self(u8::deserialize_with_mode(reader, compress, validate)?);
        if let Validate::Yes = validate {
            value.check()?;
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct FieldInlineOperandShape {
    pub op: FieldInlineOp,
    pub reads_fr_rs1: bool,
    pub reads_fr_rs2: bool,
    pub writes_fr_rd: bool,
    pub bridge_x_register_role: Option<FieldInlineXRegisterRole>,
    pub has_immediate: bool,
    /// The field destination is encoded in the `rs2` operand slot (the `rd`
    /// slot names the scratch x-register of a memory-sourced load).
    pub fr_rd_in_rs2_slot: bool,
    /// The field `rs1` read is the destination register itself (a Horner
    /// step reads the accumulator it overwrites).
    pub fr_rs1_is_fr_rd: bool,
}

impl FieldInlineOperandShape {
    /// Retain the ordinary register operands; field operands use a separate plane.
    pub fn x_operands(self, mut operands: NormalizedOperands) -> NormalizedOperands {
        operands.rs1 = match self.bridge_x_register_role {
            Some(FieldInlineXRegisterRole::ReadRs1 | FieldInlineXRegisterRole::ReadRs1WriteRd) => {
                operands.rs1
            }
            _ => None,
        };
        operands.rd = match self.bridge_x_register_role {
            Some(FieldInlineXRegisterRole::WriteRd | FieldInlineXRegisterRole::ReadRs1WriteRd) => {
                operands.rd
            }
            _ => None,
        };
        operands.rs2 = None;
        operands
    }

    pub const fn is_pure_field_op(self) -> bool {
        self.bridge_x_register_role.is_none()
    }

    pub const fn requires_product_payload(self) -> bool {
        matches!(self.op, FieldInlineOp::Mul)
    }

    pub const fn requires_inverse_product_payload(self) -> bool {
        matches!(self.op, FieldInlineOp::Inv)
    }
}

pub const fn is_field_inline_source(kind: crate::SourceInstructionKind) -> bool {
    field_inline_source_op(kind).is_some()
}

pub const fn is_field_inline_jolt(kind: crate::JoltInstructionKind) -> bool {
    field_inline_jolt_op(kind).is_some()
}

#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "fail-closed selector: non-field-inline instructions map to None"
)]
pub const fn field_inline_source_op(kind: crate::SourceInstructionKind) -> Option<FieldInlineOp> {
    match kind {
        SourceInstruction::FieldAdd(_) => Some(FieldInlineOp::Add),
        SourceInstruction::FieldSub(_) => Some(FieldInlineOp::Sub),
        SourceInstruction::FieldMul(_) => Some(FieldInlineOp::Mul),
        SourceInstruction::FieldInv(_) => Some(FieldInlineOp::Inv),
        SourceInstruction::FieldAssertEq(_) => Some(FieldInlineOp::AssertEq),
        SourceInstruction::FieldLoadFromX(_) => Some(FieldInlineOp::LoadFromX),
        SourceInstruction::FieldStoreToX(_) => Some(FieldInlineOp::StoreToX),
        SourceInstruction::FieldLoadImm(_) => Some(FieldInlineOp::LoadImm),
        SourceInstruction::FieldLoadWord(_) => Some(FieldInlineOp::LoadWord),
        SourceInstruction::FieldLoadWordHi(_) => Some(FieldInlineOp::LoadWordHi),
        SourceInstruction::FieldAdviceLimb(_) => Some(FieldInlineOp::AdviceLimb),
        _ => None,
    }
}

#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "fail-closed selector: non-field-inline instructions map to None"
)]
pub const fn field_inline_jolt_op(kind: crate::JoltInstructionKind) -> Option<FieldInlineOp> {
    match kind {
        JoltInstruction::FieldAdd(_) => Some(FieldInlineOp::Add),
        JoltInstruction::FieldSub(_) => Some(FieldInlineOp::Sub),
        JoltInstruction::FieldMul(_) => Some(FieldInlineOp::Mul),
        JoltInstruction::FieldInv(_) => Some(FieldInlineOp::Inv),
        JoltInstruction::FieldAssertEq(_) => Some(FieldInlineOp::AssertEq),
        JoltInstruction::FieldLoadFromX(_) => Some(FieldInlineOp::LoadFromX),
        JoltInstruction::FieldStoreToX(_) => Some(FieldInlineOp::StoreToX),
        JoltInstruction::FieldLoadImm(_) => Some(FieldInlineOp::LoadImm),
        JoltInstruction::FieldLoadWord(_) => Some(FieldInlineOp::LoadWord),
        JoltInstruction::FieldLoadWordHi(_) => Some(FieldInlineOp::LoadWordHi),
        JoltInstruction::FieldAdviceLimb(_) => Some(FieldInlineOp::AdviceLimb),
        _ => None,
    }
}

pub const fn field_inline_operand_shape(
    kind: crate::JoltInstructionKind,
) -> Option<FieldInlineOperandShape> {
    match field_inline_jolt_op(kind) {
        Some(op) => Some(field_inline_operand_shape_for_op(op)),
        None => None,
    }
}

pub const fn field_inline_operand_shape_for_op(op: FieldInlineOp) -> FieldInlineOperandShape {
    match op {
        FieldInlineOp::Add | FieldInlineOp::Sub | FieldInlineOp::Mul => FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: true,
            writes_fr_rd: true,
            bridge_x_register_role: None,
            has_immediate: false,
            fr_rd_in_rs2_slot: false,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::Inv => FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: None,
            has_immediate: false,
            fr_rd_in_rs2_slot: false,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::AssertEq => FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: true,
            writes_fr_rd: false,
            bridge_x_register_role: None,
            has_immediate: false,
            fr_rd_in_rs2_slot: false,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::LoadFromX => FieldInlineOperandShape {
            op,
            reads_fr_rs1: false,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::ReadRs1),
            has_immediate: false,
            fr_rd_in_rs2_slot: false,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::StoreToX => FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: false,
            writes_fr_rd: false,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::WriteRd),
            has_immediate: false,
            fr_rd_in_rs2_slot: false,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::LoadImm => FieldInlineOperandShape {
            op,
            reads_fr_rs1: false,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: None,
            has_immediate: true,
            fr_rd_in_rs2_slot: false,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::LoadWord => FieldInlineOperandShape {
            op,
            reads_fr_rs1: false,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::ReadRs1WriteRd),
            has_immediate: false,
            fr_rd_in_rs2_slot: true,
            fr_rs1_is_fr_rd: false,
        },
        FieldInlineOp::LoadWordHi => FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::ReadRs1WriteRd),
            has_immediate: false,
            fr_rd_in_rs2_slot: true,
            fr_rs1_is_fr_rd: true,
        },
        FieldInlineOp::AdviceLimb => FieldInlineOperandShape {
            op,
            reads_fr_rs1: true,
            reads_fr_rs2: false,
            writes_fr_rd: true,
            bridge_x_register_role: Some(FieldInlineXRegisterRole::WriteRd),
            has_immediate: false,
            fr_rd_in_rs2_slot: true,
            fr_rs1_is_fr_rd: false,
        },
    }
}

#[cfg(all(test, feature = "serialization"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

    fn roundtrip(
        register: FieldRegister,
        validate: Validate,
    ) -> Result<FieldRegister, SerializationError> {
        let mut bytes = Vec::new();
        register
            .serialize_with_mode(&mut bytes, Compress::No)
            .unwrap();
        FieldRegister::deserialize_with_mode(&bytes[..], Compress::No, validate)
    }

    #[test]
    fn field_register_roundtrips_in_range() {
        let register = FieldRegister(FIELD_REGISTER_COUNT - 1);
        assert_eq!(roundtrip(register, Validate::Yes).unwrap(), register);
    }

    #[test]
    fn field_register_deserialize_rejects_out_of_range() {
        // The inner field is `pub`, so an out-of-range value can be serialized directly,
        // bypassing `FieldRegister::new`; the `Valid` check must reject it on the way back.
        let register = FieldRegister(FIELD_REGISTER_COUNT);
        assert!(roundtrip(register, Validate::Yes).is_err());
        assert!(roundtrip(register, Validate::No).is_ok());
    }
}

#[cfg(test)]
mod encoding_tests {
    use super::*;

    fn r_type_word(op: FieldInlineOp, funct7: u8) -> u32 {
        u32::from(FIELD_INLINE_OPCODE)
            | (1 << 7)
            | (u32::from(op.funct3()) << 12)
            | (2 << 15)
            | (3 << 20)
            | (u32::from(funct7) << 25)
    }

    fn i_type_word(funct3: u8, imm: u16) -> u32 {
        u32::from(FIELD_INLINE_OPCODE)
            | (1 << 7)
            | (u32::from(funct3) << 12)
            | (u32::from(imm) << 20)
    }

    #[test]
    fn r_type_ops_require_exact_funct7_funct3_key() {
        assert_eq!(
            FieldInlineOp::from_r_type_key(0, 2),
            Some(FieldInlineOp::Mul)
        );
        assert_eq!(FieldInlineOp::from_r_type_key(1, 2), None);
        assert_eq!(
            FieldInlineOp::from_word(r_type_word(FieldInlineOp::Mul, 0)),
            Some(FieldInlineOp::Mul)
        );
        assert_eq!(
            FieldInlineOp::from_word(r_type_word(FieldInlineOp::Mul, 1)),
            None
        );
    }

    // Pins the encoding migration: for every op the new-format word must decode
    // back to the same `FieldInlineOp`, so the decoded op stream is unchanged.
    #[test]
    fn every_op_word_decodes_back_to_the_same_op() {
        const OPS: [FieldInlineOp; 8] = [
            FieldInlineOp::Add,
            FieldInlineOp::Sub,
            FieldInlineOp::Mul,
            FieldInlineOp::Inv,
            FieldInlineOp::AssertEq,
            FieldInlineOp::LoadFromX,
            FieldInlineOp::StoreToX,
            FieldInlineOp::LoadImm,
        ];
        for op in OPS {
            let word = match op.funct7() {
                Some(funct7) => r_type_word(op, funct7),
                None => i_type_word(op.funct3(), 0x123),
            };
            assert_eq!(FieldInlineOp::from_word(word), Some(op));
            assert_eq!(word & op.instruction_mask(), op.instruction_match());
        }
    }

    #[test]
    fn load_imm_is_reserved_i_type_family() {
        assert_eq!(
            FieldInlineOp::from_i_type_funct3(7),
            Some(FieldInlineOp::LoadImm)
        );
        assert_eq!(
            FieldInlineOp::from_word(i_type_word(7, 0x7ff)),
            Some(FieldInlineOp::LoadImm)
        );
        assert_eq!(FieldInlineOp::from_r_type_key(0, 7), None);
    }
}
