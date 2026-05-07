use super::*;

pub(in crate::expand) fn expand_scd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let v_success = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::VirtualAdvice, v_success, 0)?;

    let v_one = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::ADDI, v_one, 0, 1)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v_success, v_one, 0)?;
    asm.allocator().release(v_one)?;

    let v_addr_diff = asm.allocator().allocate()?;
    asm.emit_r(
        InstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
    )?;
    asm.emit_r(InstructionKind::MUL, v_addr_diff, v_success, v_addr_diff)?;
    asm.emit_b(InstructionKind::VirtualAssertEQ, v_addr_diff, 0, 0)?;
    asm.allocator().release(v_addr_diff)?;

    let v_mem = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::LD, v_mem, rs1(instruction)?, 0)?;

    let v_diff = asm.allocator().allocate()?;
    asm.emit_r(InstructionKind::SUB, v_diff, rs2(instruction)?, v_mem)?;
    asm.emit_r(InstructionKind::MUL, v_diff, v_diff, v_success)?;
    asm.emit_r(InstructionKind::ADD, v_diff, v_mem, v_diff)?;
    asm.allocator().release(v_mem)?;

    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v_diff, 0)?;
    asm.allocator().release(v_diff)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation, 0, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, 0, 0)?;
    asm.emit_i(InstructionKind::XORI, rd(instruction)?, v_success, 1)?;
    asm.allocator().release(v_success)?;

    asm.finalize()
}
