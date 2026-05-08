use super::*;

pub(in crate::expand) fn expand_scd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut sequence = core::ExpansionSequence::new(instruction);
    super::shared::emit_ram_region_assertion(&mut sequence, rs1(instruction)?, allocator)?;

    let v_success = allocator.allocate()?;
    sequence.emit_j_expanded(JoltInstructionKind::VirtualAdvice, v_success, 0, allocator)?;

    let v_one = allocator.allocate()?;
    sequence.emit_i_expanded(JoltInstructionKind::ADDI, v_one, 0, 1, allocator)?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertLTE,
        v_success,
        v_one,
        0,
        allocator,
    )?;
    allocator.release(v_one)?;

    let v_addr_diff = allocator.allocate()?;
    sequence.emit_r_expanded(
        JoltInstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
        allocator,
    )?;
    sequence.emit_r_expanded(
        JoltInstructionKind::MUL,
        v_addr_diff,
        v_success,
        v_addr_diff,
        allocator,
    )?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertEQ,
        v_addr_diff,
        0,
        0,
        allocator,
    )?;
    allocator.release(v_addr_diff)?;

    let v_mem = allocator.allocate()?;
    sequence.emit_i_expanded(
        JoltInstructionKind::LD,
        v_mem,
        rs1(instruction)?,
        0,
        allocator,
    )?;

    let v_diff = allocator.allocate()?;
    sequence.emit_r_expanded(
        JoltInstructionKind::SUB,
        v_diff,
        rs2(instruction)?,
        v_mem,
        allocator,
    )?;
    sequence.emit_r_expanded(
        JoltInstructionKind::MUL,
        v_diff,
        v_diff,
        v_success,
        allocator,
    )?;
    sequence.emit_r_expanded(JoltInstructionKind::ADD, v_diff, v_mem, v_diff, allocator)?;
    allocator.release(v_mem)?;

    sequence.emit_s_expanded(
        JoltInstructionKind::SD,
        rs1(instruction)?,
        v_diff,
        0,
        allocator,
    )?;
    allocator.release(v_diff)?;

    sequence.emit_i_expanded(JoltInstructionKind::ADDI, v_reservation, 0, 0, allocator)?;
    sequence.emit_i_expanded(JoltInstructionKind::ADDI, v_reservation_w, 0, 0, allocator)?;
    sequence.emit_i_expanded(
        JoltInstructionKind::XORI,
        rd(instruction)?,
        v_success,
        1,
        allocator,
    )?;

    sequence.finish_releasing(allocator, [v_success])
}
