use super::*;

pub(in crate::expand) fn expand_sll(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualPow2,
                v_pow2,
                rs2(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                rd(instruction)?,
                rs1(instruction)?,
                v_pow2,
            )),
            grammar::ExpansionOp::Release(v_pow2),
        ],
    )
}
