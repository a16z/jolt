use super::*;

pub(in crate::expand) fn expand_ecall(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    const MCAUSE_ECALL_FROM_MMODE: i128 = 11;

    let v_trap_handler_reg = allocator.trap_handler_register();
    let vr_mepc = allocator.mepc_register();
    let vr_mcause = allocator.mcause_register();
    let vr_mtval = allocator.mtval_register();
    let vr_mstatus = allocator.mstatus_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let ecall_addr = asm.allocator().allocate()?;
    asm.emit_u(JoltInstructionKind::AUIPC, ecall_addr, 0)?;
    asm.emit_i(JoltInstructionKind::ADDI, vr_mepc, ecall_addr, 0)?;
    asm.allocator().release(ecall_addr)?;

    asm.emit_i(
        JoltInstructionKind::ADDI,
        vr_mcause,
        0,
        MCAUSE_ECALL_FROM_MMODE,
    )?;
    asm.emit_i(JoltInstructionKind::ADDI, vr_mtval, 0, 0)?;

    let three = asm.allocator().allocate()?;
    asm.emit_i(JoltInstructionKind::ADDI, three, 0, 3)?;
    asm.emit_i(JoltInstructionKind::SLLI, vr_mstatus, three, 11)?;
    asm.allocator().release(three)?;

    let jalr_rd = asm.allocator().allocate()?;
    asm.emit_i(JoltInstructionKind::JALR, jalr_rd, v_trap_handler_reg, 0)?;
    asm.allocator().release(jalr_rd)?;

    asm.finalize()
}
