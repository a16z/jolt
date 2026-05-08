use super::*;

pub(in crate::expand) fn expand_sllw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        JoltInstructionKind::VirtualPow2W,
        v_pow2,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        JoltInstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    )?;
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_pow2)?;
    Ok(sequence)
}
