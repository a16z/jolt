use super::*;

pub(in crate::expand) fn expand_lwu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 4)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(InstructionKind::SRLI, rd(instruction)?, v1, 32)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}
