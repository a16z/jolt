use super::*;

pub(in crate::expand) fn expand_mret(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mepc_vr = allocator.mepc_register();
    let jalr_rd = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::JALR,
                jalr_rd,
                mepc_vr,
                0,
            )),
            grammar::ExpansionOp::Release(jalr_rd),
        ],
    )
}
