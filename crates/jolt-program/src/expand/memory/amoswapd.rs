use super::*;

pub(in crate::expand) fn expand_amoswapd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, rs2(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    Ok(sequence)
}
