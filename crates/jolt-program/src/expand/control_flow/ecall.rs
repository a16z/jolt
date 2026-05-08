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

    let mut state = core::ExpansionState::new(allocator);
    let mut sequence = core::ExpansionSequence::new(instruction);

    let ecall_addr = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
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
        ],
    )?;

    let three = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                three,
                0,
                3,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SLLI,
                vr_mstatus,
                three,
                11,
            )),
            grammar::ExpansionOp::Release(three),
        ],
    )?;

    let jalr_rd = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::JALR,
                jalr_rd,
                v_trap_handler_reg,
                0,
            )),
            grammar::ExpansionOp::Release(jalr_rd),
        ],
    )?;

    sequence.finish()
}
