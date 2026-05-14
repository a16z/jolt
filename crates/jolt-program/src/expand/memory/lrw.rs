use common::constants::RAM_START_ADDRESS;

use super::*;

/// Lowers `LR.W` by recording a word reservation and then performing `LW`.
///
/// Reservations live in dedicated virtual registers. The RAM-region assertion
/// rejects I/O addresses before the reservation is recorded, because a failed
/// store-conditional must not accidentally mutate device state through the
/// synthesized store path.
pub(in crate::expand) fn expand_lrw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation_w = reservation_w_register();
    let v_reservation_d = reservation_d_register();
    let mut asm = ExpansionBuilder::new(*instruction);
    let ram_start = asm.allocate()?;

    // LR/SC reservations are only modeled for ordinary RAM.
    asm.expand_u(
        SourceInstructionKind::LUI,
        ram_start.operand(),
        RAM_START_ADDRESS as i128,
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        ram_start.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.release(ram_start);
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(v_reservation_w),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_i(SourceInstructionKind::ADDI, reg(v_reservation_d), reg(0), 0);
    asm.expand_i(
        SourceInstructionKind::LW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        0,
    );

    asm.finalize()
}
