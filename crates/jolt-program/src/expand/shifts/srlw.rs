use super::*;

pub(in crate::expand) fn expand_srlw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let v_rs1 = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualMULI,
                v_rs1,
                rs1(instruction)?,
                1i128 << 32,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ORI,
                v_bitmask,
                rs2(instruction)?,
                32,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualShiftRightBitmask,
                v_bitmask,
                v_bitmask,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::r(
                JoltInstructionKind::VirtualSRL,
                rd(instruction)?,
                v_rs1,
                v_bitmask,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualSignExtendWord,
                rd(instruction)?,
                rd(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Release(v_bitmask),
            grammar::ExpansionOp::Release(v_rs1),
        ],
    )
}
