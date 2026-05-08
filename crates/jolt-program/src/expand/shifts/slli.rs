use super::*;

pub(in crate::expand) fn expand_slli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [grammar::ExpansionOp::Row(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualMULI,
            rd(instruction)?,
            rs1(instruction)?,
            1i128 << shift,
        ))],
    )
}
