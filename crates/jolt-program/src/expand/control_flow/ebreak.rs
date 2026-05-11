use super::*;

pub(in crate::expand) fn expand_ebreak(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    let discard = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::JAL, discard, 0)?;
    asm.allocator().release(discard)?;
    asm.finalize()
}
