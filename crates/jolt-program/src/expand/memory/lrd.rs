use common::constants::RAM_START_ADDRESS;

use super::*;

/// Lowers `LR.D` by recording a doubleword reservation and then performing `LD`.
///
/// A doubleword reservation also covers word store-conditionals at the same
/// address, so both reservation virtual registers are initialized. As with
/// `LR.W`, reservations are restricted to ordinary RAM addresses.
pub(in crate::expand) fn expand_lrd(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation_d = reservation_d_register();
    let v_reservation_w = reservation_w_register();
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
        reg(v_reservation_d),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(v_reservation_w),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::LD,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        0,
    );

    asm.finalize()
}
