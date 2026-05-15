use super::*;

/// Lowers `ECALL` into the M-mode trap-entry sequence used by Jolt.
///
/// The sequence writes the proof-facing CSR virtual registers (`mepc`,
/// `mcause`, `mtval`, `mstatus`) and then jumps to the virtual `mtvec`
/// register. Jolt's current trap model is M-mode-only with no interrupt
/// hardware, so `mstatus` is written as `MPP=M-mode, MIE=0, MPIE=0` instead of
/// using the full privileged-spec read/modify/write path.
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
    // source, matching the tracer's ECALL trap semantics.
    let ecall_addr = asm.allocate()?;
    asm.emit_u(JoltInstructionKind::AUIPC, ecall_addr.operand(), 0);
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(vr_mepc),
        ecall_addr.operand(),
        0,
    );
    asm.release(ecall_addr);
    // Machine-mode environment call, with no trap value.
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(vr_mcause),
        reg(0),
        MCAUSE_ECALL_FROM_MMODE,
    );
    asm.emit_i(JoltInstructionKind::ADDI, reg(vr_mtval), reg(0), 0);

    // 3 << 11 sets MPP=M-mode and leaves the interrupt-enable bits cleared.
    let three = asm.allocate()?;
    asm.emit_i(JoltInstructionKind::ADDI, three.operand(), reg(0), 3);
    asm.expand_i(
        SourceInstructionKind::SLLI,
        reg(vr_mstatus),
        three.operand(),
        11,
    );
    asm.release(three);

    // Jump to mtvec. The link register is a temporary because ECALL does not
    // expose a return address through the architectural register file.
    let jalr_rd = asm.allocate()?;
    asm.emit_i(
        JoltInstructionKind::JALR,
        jalr_rd.operand(),
        reg(v_trap_handler_reg),
        0,
    );
    asm.release(jalr_rd);

    asm.finalize()
}
