use super::*;

pub(in crate::expand) fn expand_divu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(JoltInstructionKind::VirtualAdvice, v0, 0)?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertValidDiv0,
        rs2(instruction)?,
        v0,
        0,
    )?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(JoltInstructionKind::MUL, v1, v0, rs2(instruction)?)?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertLTE,
        v1,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_r(JoltInstructionKind::SUB, v1, rs1(instruction)?, v1)?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v1,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}
