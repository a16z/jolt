use super::*;

pub(in crate::expand) fn expand_scd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut state = core::ExpansionState::new(allocator);
    let mut sequence = core::ExpansionSequence::new(instruction);

    let ram_start = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        super::shared::ram_region_assertion_ops(rs1(instruction)?, ram_start),
    )?;

    let v_success = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [grammar::ExpansionOp::Expand(grammar::RowTemplate::j(
            JoltInstructionKind::VirtualAdvice,
            v_success,
            0,
        ))],
    )?;

    let v_one = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v_one,
                0,
                1,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertLTE,
                v_success,
                v_one,
                0,
            )),
            grammar::ExpansionOp::Release(v_one),
        ],
    )?;

    let v_addr_diff = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SUB,
                v_addr_diff,
                v_reservation,
                rs1(instruction)?,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v_addr_diff,
                v_success,
                v_addr_diff,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertEQ,
                v_addr_diff,
                0,
                0,
            )),
            grammar::ExpansionOp::Release(v_addr_diff),
        ],
    )?;

    let v_mem = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::LD,
            v_mem,
            rs1(instruction)?,
            0,
        ))],
    )?;

    let v_diff = state.allocator().allocate()?;
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SUB,
                v_diff,
                rs2(instruction)?,
                v_mem,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v_diff,
                v_diff,
                v_success,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                v_diff,
                v_mem,
                v_diff,
            )),
            grammar::ExpansionOp::Release(v_mem),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::s(
                JoltInstructionKind::SD,
                rs1(instruction)?,
                v_diff,
                0,
            )),
            grammar::ExpansionOp::Release(v_diff),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v_reservation,
                0,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v_reservation_w,
                0,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::XORI,
                rd(instruction)?,
                v_success,
                1,
            )),
            grammar::ExpansionOp::Release(v_success),
        ],
    )?;
    sequence.finish()
}
