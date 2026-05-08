use super::*;

pub(in crate::expand) fn expand_mulhsu(
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
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualMovsign,
                v0,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ANDI,
                v1,
                v0,
                1,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::XOR,
                v2,
                rs1(instruction)?,
                v0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                v2,
                v2,
                v1,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::MULHU,
                v3,
                v2,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v2,
                v2,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::XOR,
                v3,
                v3,
                v0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::XOR,
                v2,
                v2,
                v0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                v0,
                v2,
                v1,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::SLTU,
                v0,
                v0,
                v2,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                rd(instruction)?,
                v3,
                v0,
            )),
            grammar::ExpansionOp::Release(v0),
            grammar::ExpansionOp::Release(v1),
            grammar::ExpansionOp::Release(v2),
            grammar::ExpansionOp::Release(v3),
        ],
    )
}
