use super::*;

pub(in crate::expand) fn expand_mulh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_sx = allocator.allocate()?;
    let v_sy = allocator.allocate()?;
    let v_tmp = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v_sx,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v_sy,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(JoltInstructionKind::MUL, v_sx, v_sx, rs2(instruction)?)?;
    asm.emit_r(JoltInstructionKind::MUL, v_sy, v_sy, rs1(instruction)?)?;
    asm.emit_r(
        JoltInstructionKind::MULHU,
        v_tmp,
        rs1(instruction)?,
        rs2(instruction)?,
    )?;
    asm.emit_r(JoltInstructionKind::ADD, v_tmp, v_tmp, v_sx)?;
    asm.emit_r(JoltInstructionKind::ADD, rd(instruction)?, v_tmp, v_sy)?;
    let sequence = asm.finalize()?;
    allocator.release(v_sx)?;
    allocator.release(v_sy)?;
    allocator.release(v_tmp)?;
    Ok(sequence)
}
