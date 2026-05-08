use super::*;

pub(in crate::expand) fn expand_remu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(JoltInstructionKind::VirtualAdvice, v0, 0)?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(JoltInstructionKind::MUL, v0, v0, rs2(instruction)?)?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertLTE,
        v0,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_r(JoltInstructionKind::SUB, v0, rs1(instruction)?, v0)?;
    asm.emit_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    Ok(sequence)
}
