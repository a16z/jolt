use super::*;

pub(in crate::expand) fn expand_addiw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                rs1(instruction)?,
                instruction.operands.imm,
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
