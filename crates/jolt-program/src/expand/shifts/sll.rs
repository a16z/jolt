use super::*;

pub(in crate::expand) fn expand_sll(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualPow2, v_pow2, rs2(instruction)?, 0)?;
    asm.emit_r(
        InstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_pow2)?;
    Ok(sequence)
}
