use super::*;

pub(in crate::expand) fn expand_srlw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let v_rs1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    )?;
    asm.emit_i(JoltInstructionKind::ORI, v_bitmask, rs2(instruction)?, 32)?;
    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    )?;
    asm.emit_r(
        JoltInstructionKind::VirtualSRL,
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
    allocator.release(v_bitmask)?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}
