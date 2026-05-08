use super::*;

pub(in crate::expand) fn expand_addw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                rd(instruction)?,
                rs1(instruction)?,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualSignExtendWord,
                rd(instruction)?,
                rd(instruction)?,
                0,
            )),
        ],
    )
}
