use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
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
                v2,
                v1,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SLLI,
                v0,
                v0,
                3,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ORI,
                v3,
                0,
                format_i_imm(-1),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SRLI,
                v3,
                v3,
                32,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SLL,
                v3,
                v3,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SLL,
                v0,
                rs2(instruction)?,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::XOR,
                v0,
                v2,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::AND,
                v0,
                v0,
                v3,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::XOR,
                v2,
                v2,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::s(
                JoltInstructionKind::SD,
                v1,
                v2,
                0,
            )),
            grammar::ExpansionOp::Release(v0),
            grammar::ExpansionOp::Release(v1),
            grammar::ExpansionOp::Release(v2),
            grammar::ExpansionOp::Release(v3),
        ],
    )
}
