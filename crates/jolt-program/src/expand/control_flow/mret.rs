use super::*;

pub(in crate::expand) fn expand_mret(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mepc_vr = allocator.mepc_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    let jalr_rd = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::JALR, jalr_rd, mepc_vr, 0)?;
    asm.allocator().release(jalr_rd)?;
    asm.finalize()
}
