use super::*;

pub(in crate::expand) fn expand_amoord(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_d(instruction, JoltInstructionKind::OR, allocator)
}
