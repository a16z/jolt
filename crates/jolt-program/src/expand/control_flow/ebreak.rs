use super::*;

pub(in crate::expand) fn expand_ebreak(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let discard = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_j(JoltInstructionKind::JAL, discard, 0);
    sequence.finish_releasing(allocator, [discard])
}
