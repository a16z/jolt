use super::*;

pub(in crate::expand) fn expand_amoaddd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_d(instruction, JoltInstructionKind::ADD, allocator)
}
