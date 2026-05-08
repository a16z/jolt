use super::*;

pub(in crate::expand) fn expand_lw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_align_expanded(
        JoltInstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
        allocator,
    )?;
    sequence.emit_i_expanded(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
        allocator,
    )?;
    sequence.emit_i_expanded(
        JoltInstructionKind::ANDI,
        v1,
        v0,
        format_i_imm(-8),
        allocator,
    )?;
    sequence.emit_i_expanded(JoltInstructionKind::LD, v1, v1, 0, allocator)?;
    sequence.emit_i_expanded(JoltInstructionKind::SLLI, v0, v0, 3, allocator)?;
    sequence.emit_r_expanded(JoltInstructionKind::SRL, v1, v1, v0, allocator)?;
    sequence.emit_i_expanded(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        v1,
        0,
        allocator,
    )?;
    sequence.finish_releasing(allocator, [v0, v1])
}
