#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use crate::JoltInstructionKind;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize)
)]
pub struct NormalizedOperands {
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub rd: Option<u8>,
    pub imm: i128,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize)
)]
pub struct SourceInlineKey {
    pub opcode: u8,
    pub funct3: u8,
    pub funct7: u8,
}

impl SourceInlineKey {
    #[inline]
    pub const fn packed(self) -> u32 {
        self.opcode as u32 | ((self.funct3 as u32) << 7) | ((self.funct7 as u32) << 10)
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize)
)]
pub struct SourceInstructionRow {
    pub address: usize,
    pub operands: NormalizedOperands,
    #[cfg_attr(feature = "serialization", serde(default))]
    pub inline: Option<SourceInlineKey>,
    pub is_compressed: bool,
}

impl SourceInstructionRow {
    #[inline]
    pub fn jolt_instruction_row(self, instruction_kind: JoltInstructionKind) -> JoltInstructionRow {
        let mut operands = self.operands;
        if let Some(inline) = self.inline {
            operands.imm = inline.packed() as i128;
        }
        JoltInstructionRow {
            instruction_kind,
            address: self.address,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: self.is_compressed,
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serialization",
    derive(CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize)
)]
pub struct JoltInstructionRow {
    pub instruction_kind: JoltInstructionKind,
    pub address: usize,
    pub operands: NormalizedOperands,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
