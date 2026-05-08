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
    let mut sequence = core::ExpansionSequence::new(instruction);

    let ecall_addr = allocator.allocate()?;
    sequence.emit_u(JoltInstructionKind::AUIPC, ecall_addr, 0);
    sequence.emit_i(JoltInstructionKind::ADDI, vr_mepc, ecall_addr, 0);
    allocator.release(ecall_addr)?;

    sequence.emit_i(
        JoltInstructionKind::ADDI,
        vr_mcause,
        0,
        MCAUSE_ECALL_FROM_MMODE,
    );
    sequence.emit_i(JoltInstructionKind::ADDI, vr_mtval, 0, 0);

    let three = allocator.allocate()?;
    sequence.emit_i(JoltInstructionKind::ADDI, three, 0, 3);
    sequence.emit_i(JoltInstructionKind::VirtualMULI, vr_mstatus, three, 1 << 11);
    allocator.release(three)?;

    let jalr_rd = allocator.allocate()?;
    sequence.emit_i(JoltInstructionKind::JALR, jalr_rd, v_trap_handler_reg, 0);

    sequence.finish_releasing(allocator, [jalr_rd])
}
