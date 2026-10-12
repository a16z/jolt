use jolt_riscv::SourceInstructionKind as Kind;

use super::*;

/// Lowers `SC.W` to a conditional word update whose success is
/// determined by the recorded reservation address. Returns architectural status
/// `0` on success or `1` on failure and clears both reservation registers.
pub(in crate::expand) fn expand_scw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let v_reservation = reservation_w_register();
    let v_reservation_d = reservation_d_register();
    let mut asm = ExpansionBuilder::new(*instruction);

    let v_success =
        super::shared::expand_sc_success(&mut asm, reg(rs1(instruction)?), reg(v_reservation))?;

    // Keep the success bit in the reservation register so it can gate the
    // conditional memory value below.
    asm.emit_i(Kind::ADDI, reg(v_reservation), v_success.operand(), 0);
    asm.release(v_success);

    let v_mem = asm.allocate()?;
    asm.emit_i(Kind::LW, v_mem.operand(), reg(rs1(instruction)?), 0);

    let v_diff = asm.allocate()?;
    // v_diff = old_mem + success * (rs2 - old_mem), so failure stores the
    // previous memory value and success stores rs2.
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
        reg(v_reservation),
    );
    asm.emit_r(
        Kind::ADD,
        v_diff.operand(),
        v_mem.operand(),
        v_diff.operand(),
    );
    asm.release(v_mem);

    asm.emit_i(Kind::ADDI, reg(v_reservation_d), v_diff.operand(), 0);
    asm.release(v_diff);
    asm.emit_s(Kind::SW, reg(rs1(instruction)?), reg(v_reservation_d), 0);
    asm.emit_i(Kind::XORI, reg(rd(instruction)?), reg(v_reservation), 1);
    // RISC-V invalidates the reservation after every SC, regardless of success.
    asm.emit_i(Kind::ADDI, reg(v_reservation), reg(0), 0);
    asm.emit_i(Kind::ADDI, reg(v_reservation_d), reg(0), 0);

    asm.finalize()
}
