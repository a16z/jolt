use super::*;

pub(in crate::expand) fn expand_mulhsu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualMovsign, v0, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, 1)?;
    asm.emit_r(InstructionKind::XOR, v2, rs1(instruction)?, v0)?;
    asm.emit_r(InstructionKind::ADD, v2, v2, v1)?;
    asm.emit_r(InstructionKind::MULHU, v3, v2, rs2(instruction)?)?;
    asm.emit_r(InstructionKind::MUL, v2, v2, rs2(instruction)?)?;
    asm.emit_r(InstructionKind::XOR, v3, v3, v0)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v0)?;
    asm.emit_r(InstructionKind::ADD, v0, v2, v1)?;
    asm.emit_r(InstructionKind::SLTU, v0, v0, v2)?;
    asm.emit_r(InstructionKind::ADD, rd(instruction)?, v3, v0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}
