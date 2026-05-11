use super::*;

pub(in crate::expand) fn expand_lh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_halfword_load(instruction, allocator, true)
}
