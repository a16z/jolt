use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_lrd(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation_d = reservation_d_register();
    let v_reservation_w = reservation_w_register();
    let mut asm = ExpansionBuilder::new(*instruction);
    let ram_start = asm.allocate()?;

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
        v_reservation_d,
        rs1(instruction)?,
        0,
    )?;
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v_reservation_w,
        rs1(instruction)?,
        0,
    )?;
    asm.expand_i(
        JoltInstructionKind::LD,
        rd(instruction)?,
        rs1(instruction)?,
        0,
    )?;

    asm.finalize()
}
