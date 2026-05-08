use super::*;

pub(in crate::expand) fn expand_lw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        JoltInstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(JoltInstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(JoltInstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(JoltInstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(JoltInstructionKind::SRL, v1, v1, v0)?;
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        v1,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}
