use super::*;

pub(in crate::expand) fn expand_divu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(InstructionKind::VirtualAdvice, v0, 0)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidDiv0,
        rs2(instruction)?,
        v0,
        0,
    )?;
    asm.emit_b(
        InstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(InstructionKind::MUL, v1, v0, rs2(instruction)?)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v1, rs1(instruction)?, 0)?;
    asm.emit_r(InstructionKind::SUB, v1, rs1(instruction)?, v1)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidUnsignedRemainder,
        v1,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}
