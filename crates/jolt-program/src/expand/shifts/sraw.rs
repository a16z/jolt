use super::*;

pub(in crate::expand) fn expand_sraw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        v_rs1,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(
        JoltInstructionKind::ANDI,
        v_bitmask,
        rs2(instruction)?,
        0x1f,
    )?;
    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    )?;
    asm.emit_r(
        JoltInstructionKind::VirtualSRA,
        rd(instruction)?,
        v_rs1,
        v_bitmask,
    )?;
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}
