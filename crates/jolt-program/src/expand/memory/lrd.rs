use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_lrd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_d = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let ram_start = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::u(
                JoltInstructionKind::LUI,
                ram_start,
                RAM_START_ADDRESS as i128,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertLTE,
                ram_start,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Release(ram_start),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v_reservation_d,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v_reservation_w,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                rd(instruction)?,
                rs1(instruction)?,
                0,
            )),
        ],
    )
}
