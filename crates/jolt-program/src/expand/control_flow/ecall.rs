use super::csr::MSTATUS_MIE;
use super::*;
use jolt_riscv::SourceInstructionKind as Kind;

/// Lowers `ECALL` into M-mode trap entry.
///
/// The sequence takes an environment-call-from-M-mode exception as the
/// privileged specification defines it for an M-only hart:
///
/// - `mepc := pc` of the `ECALL`;
/// - `mcause := 11`, `mtval := 0`;
/// - `mstatus.MPIE := mstatus.MIE`, `mstatus.MIE := 0` (MPP stays M, the only
///   value it can hold);
/// - `pc := mtvec.BASE`.
///
/// The `mstatus` virtual register holds only MIE and MPIE (see
/// [`super::csr`]), so the update is `mstatus := (mstatus & MIE) << 4`.
/// `mtvec` never holds a reserved MODE, so `JALR` through it, which clears
/// bit 0, lands on BASE in both Direct and Vectored mode.
pub(in crate::expand) fn expand_ecall(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    const MCAUSE_ECALL_FROM_MMODE: i128 = 11;

    let v_trap_handler_reg = trap_handler_register();
    let vr_mepc = mepc_register();
    let vr_mcause = mcause_register();
    let vr_mtval = mtval_register();
    let vr_mstatus = mstatus_register();

    let mut asm = ExpansionBuilder::new(*instruction);

    // AUIPC materializes this ECALL row's PC so mepc points back to the trap
    // source.
    asm.emit_u(Kind::AUIPC, reg(vr_mepc), 0);
    // Machine-mode environment call, with no trap value.
    asm.emit_i(Kind::ADDI, reg(vr_mcause), reg(0), MCAUSE_ECALL_FROM_MMODE);
    asm.emit_i(Kind::ADDI, reg(vr_mtval), reg(0), 0);

    // MPIE := MIE (bit 3 moves to bit 7), MIE := 0.
    let temp = asm.allocate()?;
    asm.emit_i(Kind::ANDI, temp.operand(), reg(vr_mstatus), MSTATUS_MIE);
    asm.emit_i(Kind::SLLI, reg(vr_mstatus), temp.operand(), 4);

    // Jump to mtvec. The link register is a temporary because ECALL does not
    // expose a return address through the architectural register file.
    asm.emit_i(Kind::JALR, temp.operand(), reg(v_trap_handler_reg), 0);
    asm.release(temp);

    asm.finalize()
}
