use super::*;

pub(in crate::expand) fn expand_amomaxud(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_minmax_d(instruction, JoltInstructionKind::SLTU, false, allocator)
}
