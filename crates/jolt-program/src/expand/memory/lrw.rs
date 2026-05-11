use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_lrw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation_w = reservation_w_register();
    let v_reservation_d = reservation_d_register();
    let mut asm = ExpansionBuilder::new(*instruction);
    let ram_start = asm.allocate()?;

    asm.dispatch_u(
        JoltInstructionKind::LUI,
        ram_start.operand(),
        RAM_START_ADDRESS as i128,
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertLTE,
        ram_start.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.release(ram_start);
    asm.dispatch_i(
        JoltInstructionKind::ADDI,
        reg(v_reservation_w),
        reg(rs1(instruction)?),
        0,
    );
    asm.dispatch_i(JoltInstructionKind::ADDI, reg(v_reservation_d), reg(0), 0);
    asm.dispatch_i(
        JoltInstructionKind::LW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        0,
    );

    asm.finalize()
}
