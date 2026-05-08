use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
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
    sequence.emit_i_expanded(JoltInstructionKind::LD, v2, v1, 0, allocator)?;
    sequence.emit_i_expanded(JoltInstructionKind::SLLI, v0, v0, 3, allocator)?;
    sequence.emit_i_expanded(JoltInstructionKind::ORI, v3, 0, format_i_imm(-1), allocator)?;
    sequence.emit_i_expanded(JoltInstructionKind::SRLI, v3, v3, 32, allocator)?;
    sequence.emit_r_expanded(JoltInstructionKind::SLL, v3, v3, v0, allocator)?;
    sequence.emit_r_expanded(
        JoltInstructionKind::SLL,
        v0,
        rs2(instruction)?,
        v0,
        allocator,
    )?;
    sequence.emit_r_expanded(JoltInstructionKind::XOR, v0, v2, v0, allocator)?;
    sequence.emit_r_expanded(JoltInstructionKind::AND, v0, v0, v3, allocator)?;
    sequence.emit_r_expanded(JoltInstructionKind::XOR, v2, v2, v0, allocator)?;
    sequence.emit_s_expanded(JoltInstructionKind::SD, v1, v2, 0, allocator)?;
    sequence.finish_releasing(allocator, [v0, v1, v2, v3])
}
