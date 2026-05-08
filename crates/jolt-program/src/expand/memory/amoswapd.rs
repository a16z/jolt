use super::*;

pub(in crate::expand) fn expand_amoswapd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                v_rd,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::s(
                JoltInstructionKind::SD,
                rs1(instruction)?,
                rs2(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                v_rd,
                0,
            )),
            grammar::ExpansionOp::Release(v_rd),
        ],
    )
}
