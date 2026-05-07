use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
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
    asm.emit_i(InstructionKind::LD, v2, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_i(InstructionKind::ORI, v3, 0, format_i_imm(-1))?;
    asm.emit_i(InstructionKind::SRLI, v3, v3, 32)?;
    asm.emit_r(InstructionKind::SLL, v3, v3, v0)?;
    asm.emit_r(InstructionKind::SLL, v0, rs2(instruction)?, v0)?;
    asm.emit_r(InstructionKind::XOR, v0, v2, v0)?;
    asm.emit_r(InstructionKind::AND, v0, v0, v3)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v0)?;
    asm.emit_s(InstructionKind::SD, v1, v2, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}
