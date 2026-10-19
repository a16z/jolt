#[cfg(feature = "serialization")]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "field-inline")]
use crate::field_inline_operand_shape;
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

impl JoltInstructionRow {
    /// The row fields its decoded flags depend on.
    ///
    /// [`jolt_instruction!`](crate::jolt_instruction) derives every
    /// instruction's circuit and instruction flags from its kind plus the
    /// virtual-sequence state, `is_compressed`, and `is_first_in_sequence`, and
    /// the kind alone fixes its instruction type. Rows with equal classes
    /// therefore decode to the same flags, whatever their address and operands.
    ///
    /// WARNING: the verifier's folded read-RAF evaluation reads each class's
    /// flags from one row, so a flag that depended on an excluded field would
    /// break soundness. A field added to the row fails to compile here until
    /// it is keyed or excluded; only `flag_class_determines_read_raf_flag_terms`
    /// guards a flag starting to read the address or operands.
    pub fn flag_class(&self) -> u32 {
        // Names every field without `..`, so a new row field must be decided here.
        let Self {
            instruction_kind,
            virtual_sequence_remaining,
            is_compressed,
            is_first_in_sequence,
            address: _,
            operands: _,
        } = self;
        let sequence = match virtual_sequence_remaining {
            None => 0,
            Some(0) => 1,
            Some(_) => 2,
        };
        (u32::from(instruction_kind.tag().0) << 4)
            | (sequence << 2)
            | (u32::from(*is_compressed) << 1)
            | u32::from(*is_first_in_sequence)
    }

    /// Logical operands belonging to the field register file, including an
    /// accumulator's implicit read and bridge destinations encoded in `rs2`.
    pub fn field_operands(&self) -> NormalizedOperands {
        #[cfg(feature = "field-inline")]
        if let Some(shape) = field_inline_operand_shape(self.instruction_kind) {
            return shape.field_operands(self.operands);
        }
        NormalizedOperands::default()
    }

    /// Operands belonging to the integer register file. Field-register slots
    /// are absent; bridge instructions retain their integer source or destination.
    pub fn integer_operands(&self) -> NormalizedOperands {
        #[cfg(feature = "field-inline")]
        if let Some(shape) = field_inline_operand_shape(self.instruction_kind) {
            return shape.x_operands(self.operands);
        }
        self.operands
    }
}
