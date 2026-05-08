use super::*;

pub(in crate::expand) fn expand_srl(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualShiftRightBitmask,
                v_bitmask,
                rs2(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::VirtualSRL,
                rd(instruction)?,
                rs1(instruction)?,
                v_bitmask,
            )),
            grammar::ExpansionOp::Release(v_bitmask),
        ],
    )
}
