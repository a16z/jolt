use jolt_riscv::SourceInstructionKind as Kind;

use super::*;

/// Lowers `SC.D` to a conditional doubleword update whose success is
/// determined by the recorded reservation address. Returns architectural status
/// `0` on success or `1` on failure and clears both reservation registers.
pub(in crate::expand) fn expand_scd(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation = reservation_d_register();
    let v_reservation_w = reservation_w_register();
    let mut asm = ExpansionBuilder::new(*instruction);

    let v_success =
        super::shared::expand_sc_success(&mut asm, reg(rs1(instruction)?), reg(v_reservation))?;

    let v_mem = asm.allocate()?;
    asm.emit_i(Kind::LD, v_mem.operand(), reg(rs1(instruction)?), 0);

    let v_diff = asm.allocate()?;
    asm.emit_r(
        Kind::SUB,
        v_diff.operand(),
        reg(rs2(instruction)?),
        v_mem.operand(),
    );
    asm.emit_r(
        Kind::MUL,
        v_diff.operand(),
        v_diff.operand(),
        v_success.operand(),
    );
    asm.emit_r(
        Kind::ADD,
        v_diff.operand(),
        v_mem.operand(),
        v_diff.operand(),
    );
    asm.release(v_mem);
    asm.emit_s(Kind::SD, reg(rs1(instruction)?), v_diff.operand(), 0);
    asm.release(v_diff);
    // RISC-V invalidates the reservation after every SC, regardless of success.
    asm.emit_i(Kind::ADDI, reg(v_reservation), reg(0), 0);
    asm.emit_i(Kind::ADDI, reg(v_reservation_w), reg(0), 0);
    asm.emit_i(Kind::XORI, reg(rd(instruction)?), v_success.operand(), 1);
    asm.release(v_success);

    asm.finalize()
}
