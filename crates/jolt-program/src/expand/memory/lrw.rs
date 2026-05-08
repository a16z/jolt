use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_lrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_w = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let ram_start = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.expand_u(
        JoltInstructionKind::LUI,
        ram_start,
        RAM_START_ADDRESS as i128,
    )?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertLTE,
        ram_start,
        rs1(instruction)?,
        0,
    )?;
    asm.release(ram_start)?;
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v_reservation_w,
        rs1(instruction)?,
        0,
    )?;
    asm.expand_i(JoltInstructionKind::ADDI, v_reservation_d, 0, 0)?;
    asm.expand_i(
        JoltInstructionKind::LW,
        rd(instruction)?,
        rs1(instruction)?,
        0,
    )?;

    asm.finalize()
}
