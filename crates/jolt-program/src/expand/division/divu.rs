use super::*;

pub(in crate::expand) fn expand_divu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_j_expanded(JoltInstructionKind::VirtualAdvice, v0, 0, allocator)?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertValidDiv0,
        rs2(instruction)?,
        v0,
        0,
        allocator,
    )?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_r_expanded(
        JoltInstructionKind::MUL,
        v1,
        v0,
        rs2(instruction)?,
        allocator,
    )?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertLTE,
        v1,
        rs1(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_r_expanded(
        JoltInstructionKind::SUB,
        v1,
        rs1(instruction)?,
        v1,
        allocator,
    )?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v1,
        rs2(instruction)?,
        0,
        allocator,
    )?;
    sequence.emit_i_expanded(
        JoltInstructionKind::ADDI,
        rd(instruction)?,
        v0,
        0,
        allocator,
    )?;
    sequence.finish_releasing(allocator, [v0, v1])
}
