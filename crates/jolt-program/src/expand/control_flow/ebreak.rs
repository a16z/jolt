use super::*;

pub(in crate::expand) fn expand_ebreak(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let discard = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.emit_j(JoltInstructionKind::JAL, discard, 0);
    asm.release(discard)?;

    asm.finalize()
}
