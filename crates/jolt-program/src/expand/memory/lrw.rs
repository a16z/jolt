use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_lrw(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation_w = reservation_w_register();
    let v_reservation_d = reservation_d_register();
    let mut asm = ExpansionBuilder::new(*instruction);
    let ram_start = asm.allocate()?;

    asm.expand_u(
        JoltInstructionKind::LUI,
        ram_start.operand(),
        RAM_START_ADDRESS as i128,
    );
    asm.expand_b(
        JoltInstructionKind::VirtualAssertLTE,
        ram_start.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.release(ram_start);
    asm.expand_i(
        JoltInstructionKind::ADDI,
        reg(v_reservation_w),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_i(JoltInstructionKind::ADDI, reg(v_reservation_d), reg(0), 0);
    asm.expand_i(
        JoltInstructionKind::LW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        0,
    );

    asm.finalize()
}
