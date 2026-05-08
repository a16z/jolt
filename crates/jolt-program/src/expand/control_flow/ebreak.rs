use super::*;

pub(in crate::expand) fn expand_ebreak(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let discard = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::j(
                JoltInstructionKind::JAL,
                discard,
                0,
            )),
            grammar::ExpansionOp::Release(discard),
        ],
    )
}
