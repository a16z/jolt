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

    let ecall_addr = allocator.allocate()?;
    let three = allocator.allocate()?;
    let jalr_rd = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::u(
                JoltInstructionKind::AUIPC,
                ecall_addr,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                vr_mepc,
                ecall_addr,
                0,
            )),
            grammar::ExpansionOp::Release(ecall_addr),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                vr_mcause,
                0,
                MCAUSE_ECALL_FROM_MMODE,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                vr_mtval,
                0,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                three,
                0,
                3,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualMULI,
                vr_mstatus,
                three,
                1 << 11,
            )),
            grammar::ExpansionOp::Release(three),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::JALR,
                jalr_rd,
                v_trap_handler_reg,
                0,
            )),
            grammar::ExpansionOp::Release(jalr_rd),
        ],
    )
}
