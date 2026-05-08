use super::*;

pub(in crate::expand) fn expand_remu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::j(
                JoltInstructionKind::VirtualAdvice,
                v0,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertMulUNoOverflow,
                v0,
                rs2(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v0,
                v0,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertLTE,
                v0,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SUB,
                v0,
                rs1(instruction)?,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
                v0,
                rs2(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                v0,
                0,
            )),
            grammar::ExpansionOp::Release(v0),
        ],
    )
}
