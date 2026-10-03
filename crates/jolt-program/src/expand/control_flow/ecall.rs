use super::*;

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
    asm.emit_u(Kind::AUIPC, ecall_addr.operand(), 0);
    asm.emit_i(Kind::ADDI, reg(vr_mepc), ecall_addr.operand(), 0);
    asm.release(ecall_addr);
    asm.emit_i(Kind::ADDI, reg(vr_mcause), reg(0), MCAUSE_ECALL_FROM_MMODE);
    asm.emit_i(Kind::ADDI, reg(vr_mtval), reg(0), 0);

    let three = asm.allocate()?;
    asm.emit_i(Kind::ADDI, three.operand(), reg(0), 3);
    asm.emit_i(
        SourceInstructionKind::SLLI,
        reg(vr_mstatus),
        three.operand(),
        11,
    );
    asm.release(three);

    let jalr_rd = asm.allocate()?;
    asm.emit_i(Kind::JALR, jalr_rd.operand(), reg(v_trap_handler_reg), 0);
    asm.release(jalr_rd);

    asm.finalize()
}
