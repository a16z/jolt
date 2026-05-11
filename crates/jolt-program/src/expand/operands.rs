use super::*;

pub(super) fn noop_for_source(instruction: SourceInstruction) -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind: JoltInstructionKind::ADDI,
        address: instruction.address,
        operands: NormalizedOperands {
            rd: Some(0),
            rs1: Some(0),
            rs2: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: instruction.is_compressed,
    }
}

/// Replaces a side-effect-free rd=x0 instruction with `ADDI x0, x0, 0`.
pub(super) fn noop_for(instruction: NormalizedInstruction) -> NormalizedInstruction {
    debug_assert_eq!(instruction.operands.rd, Some(0));
    NormalizedInstruction {
        instruction_kind: JoltInstructionKind::ADDI,
        address: instruction.address,
        operands: NormalizedOperands {
            rd: Some(0),
            rs1: Some(0),
            rs2: None,
            imm: 0,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: instruction.is_compressed,
    }
}

pub(super) fn rd(instruction: &NormalizedInstruction) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rd
        .ok_or(ExpansionError::MalformedInstruction("missing rd"))
}

pub(super) fn rs1(instruction: &NormalizedInstruction) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rs1
        .ok_or(ExpansionError::MalformedInstruction("missing rs1"))
}

pub(super) fn rs2(instruction: &NormalizedInstruction) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rs2
        .ok_or(ExpansionError::MalformedInstruction("missing rs2"))
}

pub(super) fn format_i_imm(imm: i128) -> i128 {
    (imm as i64 as u64) as i128
}

pub(super) fn csr_address(instruction: &NormalizedInstruction) -> u16 {
    (instruction.operands.imm & 0xfff) as u16
}

/// Final rows whose expansion dispatch handles rd=x0 itself.
pub(super) const fn handles_final_rd_zero_internally(
    instruction_kind: JoltInstructionKind,
) -> bool {
    matches!(
        instruction_kind,
        JoltInstructionKind::ECALL
            | JoltInstructionKind::MRET
            | JoltInstructionKind::EBREAK
            | JoltInstructionKind::CSRRW
            | JoltInstructionKind::CSRRS
    )
}
