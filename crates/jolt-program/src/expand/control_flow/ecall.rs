use super::*;

pub(in crate::expand) fn expand_ecall(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    const MCAUSE_ECALL_FROM_MMODE: i128 = 11;

    let v_trap_handler_reg = trap_handler_register();
    let vr_mepc = mepc_register();
    let vr_mcause = mcause_register();
    let vr_mtval = mtval_register();
    let vr_mstatus = mstatus_register();

    let mut asm = ExpansionBuilder::new(*instruction);

    let ecall_addr = asm.allocate()?;
    asm.emit_u(JoltInstructionKind::AUIPC, ecall_addr.operand(), 0);
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(vr_mepc),
        ecall_addr.operand(),
        0,
    );
    asm.release(ecall_addr)?;
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(vr_mcause),
        reg(0),
        MCAUSE_ECALL_FROM_MMODE,
    );
    asm.emit_i(JoltInstructionKind::ADDI, reg(vr_mtval), reg(0), 0);

    let three = asm.allocate()?;
    asm.emit_i(JoltInstructionKind::ADDI, three.operand(), reg(0), 3);
    asm.expand_i(
        JoltInstructionKind::SLLI,
        reg(vr_mstatus),
        three.operand(),
        11,
    )?;
    asm.release(three)?;

    let jalr_rd = asm.allocate()?;
    asm.emit_i(
        JoltInstructionKind::JALR,
        jalr_rd.operand(),
        reg(v_trap_handler_reg),
        0,
    );
    asm.release(jalr_rd)?;

    asm.finalize()
}
