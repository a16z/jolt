use super::*;

pub(in crate::expand) fn expand_sra(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRA,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}
