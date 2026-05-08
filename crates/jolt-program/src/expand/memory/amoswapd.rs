use super::*;

pub(in crate::expand) fn expand_amoswapd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i_expanded(
        JoltInstructionKind::LD,
        v_rd,
        rs1(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_s_expanded(
        JoltInstructionKind::SD,
        rs1(instruction)?,
        rs2(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_i_expanded(
        JoltInstructionKind::ADDI,
        rd(instruction)?,
        v_rd,
        0,
        allocator,
    )?;
    sequence.finish_releasing(allocator, [v_rd])
}
