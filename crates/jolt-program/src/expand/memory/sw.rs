use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    asm.expand_address(
        JoltInstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.expand_i(JoltInstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.expand_i(JoltInstructionKind::LD, v2, v1, 0)?;
    asm.expand_i(JoltInstructionKind::SLLI, v0, v0, 3)?;
    asm.expand_i(JoltInstructionKind::ORI, v3, 0, format_i_imm(-1))?;
    asm.expand_i(JoltInstructionKind::SRLI, v3, v3, 32)?;
    asm.expand_r(JoltInstructionKind::SLL, v3, v3, v0)?;
    asm.expand_r(JoltInstructionKind::SLL, v0, rs2(instruction)?, v0)?;
    asm.expand_r(JoltInstructionKind::XOR, v0, v2, v0)?;
    asm.expand_r(JoltInstructionKind::AND, v0, v0, v3)?;
    asm.expand_r(JoltInstructionKind::XOR, v2, v2, v0)?;
    asm.expand_s(JoltInstructionKind::SD, v1, v2, 0)?;
    asm.release_many([v0, v1, v2, v3])?;

    asm.finalize()
}
