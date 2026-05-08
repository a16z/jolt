use super::*;

pub(in crate::expand) fn expand_mulh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_sx = allocator.allocate()?;
    let v_sy = allocator.allocate()?;
    let v_tmp = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualMovsign,
                v_sx,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualMovsign,
                v_sy,
                rs2(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v_sx,
                v_sx,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v_sy,
                v_sy,
                rs1(instruction)?,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::MULHU,
                v_tmp,
                rs1(instruction)?,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                v_tmp,
                v_tmp,
                v_sx,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                rd(instruction)?,
                v_tmp,
                v_sy,
            )),
            grammar::ExpansionOp::Release(v_sx),
            grammar::ExpansionOp::Release(v_sy),
            grammar::ExpansionOp::Release(v_tmp),
        ],
    )
}
