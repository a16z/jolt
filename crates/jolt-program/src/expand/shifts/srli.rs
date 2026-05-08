use super::*;

pub(in crate::expand) fn expand_srli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [grammar::ExpansionOp::Row(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualSRLI,
            rd(instruction)?,
            rs1(instruction)?,
            bitmask as i128,
        ))],
    )
}
