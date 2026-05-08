use super::*;

pub(in crate::expand) fn expand_lwu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

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
    asm.expand_i(JoltInstructionKind::LD, v1, v1, 0)?;
    asm.expand_i(JoltInstructionKind::XORI, v0, v0, 4)?;
    asm.expand_i(JoltInstructionKind::SLLI, v0, v0, 3)?;
    asm.expand_r(JoltInstructionKind::SLL, v1, v1, v0)?;
    asm.expand_i(JoltInstructionKind::SRLI, rd(instruction)?, v1, 32)?;
    asm.release(v0)?;
    asm.release(v1)?;

    asm.finalize()
}
