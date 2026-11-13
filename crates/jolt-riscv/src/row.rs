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
            source_can_end: false,
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
    /// Whether the source of an expanded row permits stopping at its terminal jump.
    #[cfg_attr(feature = "serialization", serde(default))]
    pub source_can_end: bool,
}

impl JoltInstructionRow {
    /// Whether this row may end a trace after jumping to its own address.
    #[inline]
    pub fn can_end(&self) -> bool {
        let repeatable_jump = matches!(
            self.instruction_kind,
            JoltInstructionKind::Jal(_) | JoltInstructionKind::Jalr(_)
        ) && self.self_jump_repeats();
        repeatable_jump
            && match self.virtual_sequence_remaining {
                None => true,
                Some(0) => self.source_can_end,
                Some(_) => false,
            }
    }

    /// The row fields its decoded flags depend on.
    ///
    /// [`jolt_instruction!`](crate::jolt_instruction) derives every
    /// instruction's circuit and instruction flags from its kind plus the
    /// virtual-sequence state, `is_compressed`, and `is_first_in_sequence`, and
    /// the kind alone fixes its instruction type. Rows with equal classes
    /// therefore decode to the same flags. `CanEnd` additionally depends on
    /// the source classification and repeatability of the jump operands.
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
            source_can_end: _,
        } = self;
        let sequence = match virtual_sequence_remaining {
            None => 0,
            Some(0) => 1,
            Some(_) => 2,
        };
        (u32::from(instruction_kind.tag().0) << 5)
            | (sequence << 2)
            | (u32::from(*is_compressed) << 1)
            | (u32::from(self.can_end()) << 4)
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

    /// Whether this row, having just jumped to its own address, jumps there
    /// again when re-run: the PC-stall fixed point at which tracers stop. The
    /// tracers reject a Jolt run that stops here instead of at a halt
    /// (`TraceError::EndedWithoutHalt`). Only a `Jalr` that overwrites its own
    /// base register can leave, since its re-run targets `(link + imm) & !1`.
    pub fn self_jump_repeats(&self) -> bool {
        let JoltInstructionKind::Jalr(_) = self.instruction_kind else {
            return true;
        };
        let NormalizedOperands {
            rd: Some(rd),
            rs1: Some(rs1),
            imm,
            ..
        } = self.operands
        else {
            return true;
        };
        if rd == 0 || rd != rs1 {
            return true;
        }
        let address = self.address as u64;
        let link = address.wrapping_add(if self.is_compressed { 2 } else { 4 });
        link.wrapping_add(imm as u64) & !1 == address
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instructions::{Jal, Jalr},
        CircuitFlags, Flags, JoltInstruction,
    };

    #[test]
    fn can_end_requires_a_repeatable_source_jump() {
        for (kind, rd, rs1, imm, compressed, expected) in [
            (JoltInstructionKind::JAL, 0, 0, 0, false, true),
            (JoltInstructionKind::JAL, 0, 0, 0, true, true),
            (JoltInstructionKind::JALR, 5, 5, 0, false, false),
            (JoltInstructionKind::JALR, 1, 1, 0, true, false),
            (JoltInstructionKind::JALR, 0, 5, 0, false, true),
            (JoltInstructionKind::JALR, 5, 5, -4, false, true),
            (JoltInstructionKind::JALR, 5, 5, -3, false, true),
            (JoltInstructionKind::JALR, 1, 1, -2, true, true),
            (JoltInstructionKind::JALR, 1, 1, -4, true, false),
            (JoltInstructionKind::BEQ, 0, 0, 0, false, false),
        ] {
            let row = JoltInstructionRow {
                instruction_kind: kind,
                address: 0x8000_0000,
                operands: NormalizedOperands {
                    rd: Some(rd),
                    rs1: Some(rs1),
                    imm,
                    ..Default::default()
                },
                is_compressed: compressed,
                ..Default::default()
            };
            assert_eq!(row.can_end(), expected, "{row:?}");
            assert_eq!(
                JoltInstruction::from(row).circuit_flags()[CircuitFlags::CanEnd],
                expected,
                "{row:?}"
            );
        }
    }

    #[test]
    fn typed_jump_flags_use_the_instruction_variant() {
        let row = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            operands: NormalizedOperands {
                rs1: Some(5),
                rd: Some(5),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(Jal(row).circuit_flags()[CircuitFlags::CanEnd]);
        assert!(!Jalr(row).circuit_flags()[CircuitFlags::CanEnd]);
        let stable = JoltInstructionRow {
            operands: NormalizedOperands {
                imm: -4,
                ..row.operands
            },
            ..row
        };
        assert!(Jalr(stable).circuit_flags()[CircuitFlags::CanEnd]);
    }

    #[test]
    fn expanded_jump_needs_source_permission_and_terminal_position() {
        let mut row = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::JAL,
            virtual_sequence_remaining: Some(0),
            ..Default::default()
        };
        assert!(!row.can_end());
        row.source_can_end = true;
        assert!(row.can_end());
        row.virtual_sequence_remaining = Some(1);
        assert!(!row.can_end());
        row.virtual_sequence_remaining = Some(0);
        row.instruction_kind = JoltInstructionKind::ADDI;
        assert!(!row.can_end());
    }
}
