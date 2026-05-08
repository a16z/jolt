use super::*;

pub(in crate::expand) fn expand_scw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    let ram_start = asm.allocate()?;
    super::shared::emit_ram_region_assertion(&mut asm, rs1(instruction)?, ram_start)?;

    let v_success = asm.allocate()?;
    asm.expand_j(JoltInstructionKind::VirtualAdvice, v_success, 0)?;

    let v_one = asm.allocate()?;
    asm.expand_i(JoltInstructionKind::ADDI, v_one, 0, 1)?;
    asm.expand_b(JoltInstructionKind::VirtualAssertLTE, v_success, v_one, 0)?;
    asm.release(v_one)?;

    let v_addr_diff = asm.allocate()?;
    asm.expand_r(
        JoltInstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
    )?;
    asm.expand_r(
        JoltInstructionKind::MUL,
        v_addr_diff,
        v_success,
        v_addr_diff,
    )?;
    asm.expand_b(JoltInstructionKind::VirtualAssertEQ, v_addr_diff, 0, 0)?;
    asm.release(v_addr_diff)?;

    asm.expand_i(JoltInstructionKind::ADDI, v_reservation, v_success, 0)?;
    asm.release(v_success)?;

    let v_mem = asm.allocate()?;
    asm.expand_i(JoltInstructionKind::LW, v_mem, rs1(instruction)?, 0)?;

    let v_diff = asm.allocate()?;
    asm.expand_r(JoltInstructionKind::SUB, v_diff, rs2(instruction)?, v_mem)?;
    asm.expand_r(JoltInstructionKind::MUL, v_diff, v_diff, v_reservation)?;
    asm.expand_r(JoltInstructionKind::ADD, v_diff, v_mem, v_diff)?;
    asm.release(v_mem)?;

    asm.expand_i(JoltInstructionKind::ADDI, v_reservation_d, v_diff, 0)?;
    asm.release(v_diff)?;
    asm.expand_s(
        JoltInstructionKind::SW,
        rs1(instruction)?,
        v_reservation_d,
        0,
    )?;
    asm.expand_i(
        JoltInstructionKind::XORI,
        rd(instruction)?,
        v_reservation,
        1,
    )?;
    asm.expand_i(JoltInstructionKind::ADDI, v_reservation, 0, 0)?;
    asm.expand_i(JoltInstructionKind::ADDI, v_reservation_d, 0, 0)?;

    asm.finalize()
}
