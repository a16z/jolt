use super::*;

pub(super) fn noop_for(instruction: NormalizedInstruction) -> NormalizedInstruction {
    debug_assert_eq!(instruction.operands.rd, Some(0));
    NormalizedInstruction {
        instruction_kind: InstructionKind::ADDI,
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

pub(super) const fn handles_rd_zero_internally(instruction_kind: InstructionKind) -> bool {
    matches!(
        instruction_kind,
        InstructionKind::ECALL
            | InstructionKind::MRET
            | InstructionKind::EBREAK
            | InstructionKind::CSRRW
            | InstructionKind::CSRRS
    )
}

pub(super) const fn has_side_effects(instruction_kind: InstructionKind) -> bool {
    matches!(
        instruction_kind,
        InstructionKind::AdviceLB
            | InstructionKind::AdviceLD
            | InstructionKind::AdviceLH
            | InstructionKind::AdviceLW
            | InstructionKind::AMOADDD
            | InstructionKind::AMOADDW
            | InstructionKind::AMOANDD
            | InstructionKind::AMOANDW
            | InstructionKind::AMOMAXD
            | InstructionKind::AMOMAXUD
            | InstructionKind::AMOMAXUW
            | InstructionKind::AMOMAXW
            | InstructionKind::AMOMIND
            | InstructionKind::AMOMINUD
            | InstructionKind::AMOMINUW
            | InstructionKind::AMOMINW
            | InstructionKind::AMOORD
            | InstructionKind::AMOORW
            | InstructionKind::AMOSWAPD
            | InstructionKind::AMOSWAPW
            | InstructionKind::AMOXORD
            | InstructionKind::AMOXORW
            | InstructionKind::BEQ
            | InstructionKind::BGE
            | InstructionKind::BGEU
            | InstructionKind::BLT
            | InstructionKind::BLTU
            | InstructionKind::BNE
            | InstructionKind::CSRRS
            | InstructionKind::CSRRW
            | InstructionKind::EBREAK
            | InstructionKind::ECALL
            | InstructionKind::Inline
            | InstructionKind::JAL
            | InstructionKind::JALR
            | InstructionKind::LB
            | InstructionKind::LBU
            | InstructionKind::LD
            | InstructionKind::LH
            | InstructionKind::LHU
            | InstructionKind::LRD
            | InstructionKind::LRW
            | InstructionKind::LW
            | InstructionKind::LWU
            | InstructionKind::MRET
            | InstructionKind::SB
            | InstructionKind::SCD
            | InstructionKind::SCW
            | InstructionKind::SD
            | InstructionKind::SH
            | InstructionKind::SW
            | InstructionKind::VirtualAdviceLoad
            | InstructionKind::VirtualHostIO
            | InstructionKind::VirtualSW
    )
}
