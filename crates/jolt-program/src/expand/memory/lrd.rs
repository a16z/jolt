use super::*;

pub(in crate::expand) fn expand_lrd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_d = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::LD, rd(instruction)?, rs1(instruction)?, 0)?;
    asm.finalize()
}
