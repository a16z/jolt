use super::*;

pub(in crate::expand) fn expand_lrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_w = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    super::shared::emit_ram_region_assertion(&mut asm, rs1(instruction)?)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, 0, 0)?;
    asm.emit_i(InstructionKind::LW, rd(instruction)?, rs1(instruction)?, 0)?;
    asm.finalize()
}
