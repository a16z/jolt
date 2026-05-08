use super::*;

pub(in crate::expand) fn expand_amoswapd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.expand_i(JoltInstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.expand_s(
        JoltInstructionKind::SD,
        rs1(instruction)?,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    asm.release(v_rd)?;

    asm.finalize()
}
