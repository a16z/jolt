#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use crate::{JoltInstructionKind, SourceInstructionKind};

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

/// Instruction row decoded from guest program text before bytecode expansion.
///
/// A source row represents what the guest program asked to execute. It may be a
/// standard RV64 instruction or a Jolt custom source opcode such as a registered
/// inline or advice load. Expansion maps source rows into final
/// [`NormalizedInstruction`] bytecode rows.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct SourceInstruction {
    pub instruction_kind: SourceInstructionKind,
    pub address: usize,
    pub operands: NormalizedOperands,
    pub is_compressed: bool,
}

impl SourceInstruction {
    /// Converts this decoded source row into the current normalized row shape.
    ///
    /// This is the explicit bridge used while `NormalizedInstruction` remains
    /// the final bytecode row type. Source rows do not carry virtual-sequence
    /// metadata; expansion assigns that metadata on the emitted bytecode rows.
    pub fn into_normalized_instruction(self) -> NormalizedInstruction {
        NormalizedInstruction {
            instruction_kind: self.instruction_kind.jolt_kind(),
            address: self.address,
            operands: self.operands,
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
pub struct NormalizedInstruction {
    pub instruction_kind: JoltInstructionKind,
    pub address: usize,
    pub operands: NormalizedOperands,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}
