use super::*;

/// Replaces a side-effect-free rd=x0 instruction with `ADDI x0, x0, 0`.
pub(super) fn noop_for(instruction: SourceInstructionRow) -> JoltInstructionRow {
    debug_assert_eq!(instruction.operands.rd, Some(0));
    JoltInstructionRow {
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

pub(super) fn rd(instruction: &SourceInstructionRow) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rd
        .ok_or(ExpansionError::MalformedInstruction("missing rd"))
}

pub(super) fn rs1(instruction: &SourceInstructionRow) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rs1
        .ok_or(ExpansionError::MalformedInstruction("missing rs1"))
}

pub(super) fn rs2(instruction: &SourceInstructionRow) -> Result<u8, ExpansionError> {
    instruction
        .operands
        .rs2
        .ok_or(ExpansionError::MalformedInstruction("missing rs2"))
}

pub(super) fn format_i_imm(imm: i128) -> i128 {
    (imm as i64 as u64) as i128
}

pub(super) fn csr_address(instruction: &SourceInstructionRow) -> u16 {
    (instruction.operands.imm & 0xfff) as u16
}

/// Instructions whose expansion recipes handle rd=x0 themselves (trap, CSR).
pub(super) const fn handles_rd_zero_internally(instruction_kind: SourceInstructionKind) -> bool {
    matches!(
        instruction_kind,
        SourceInstructionKind::ECALL
            | SourceInstructionKind::MRET
            | SourceInstructionKind::EBREAK
            | SourceInstructionKind::CSRRW
            | SourceInstructionKind::CSRRS
    )
}
