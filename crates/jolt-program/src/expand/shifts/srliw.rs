use super::*;

pub(in crate::expand) fn expand_srliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = (instruction.operands.imm & 0x1f) + 32;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
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
                JoltInstructionKind::VirtualSRLI,
                rd(instruction)?,
                v_rs1,
                bitmask as i128,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualSignExtendWord,
                rd(instruction)?,
                rd(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Release(v_rs1),
        ],
    )
}
