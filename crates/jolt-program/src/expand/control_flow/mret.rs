use super::*;

pub(in crate::expand) fn expand_mret(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mepc_vr = allocator.mepc_register();
    let jalr_rd = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(JoltInstructionKind::JALR, jalr_rd, mepc_vr, 0);
    sequence.finish_releasing(allocator, [jalr_rd])
}
