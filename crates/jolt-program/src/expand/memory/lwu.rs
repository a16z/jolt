use super::*;

pub(in crate::expand) fn expand_lwu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::address(
                JoltInstructionKind::VirtualAssertWordAlignment,
                rs1(instruction)?,
                instruction.operands.imm,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v0,
                rs1(instruction)?,
                format_i_imm(instruction.operands.imm),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ANDI,
                v1,
                v0,
                format_i_imm(-8),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                v1,
                v1,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::XORI,
                v0,
                v0,
                4,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SLLI,
                v0,
                v0,
                3,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SLL,
                v1,
                v1,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SRLI,
                rd(instruction)?,
                v1,
                32,
            )),
            grammar::ExpansionOp::Release(v0),
            grammar::ExpansionOp::Release(v1),
        ],
    )
}
