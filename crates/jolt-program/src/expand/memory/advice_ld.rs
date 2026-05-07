use super::*;

pub(in crate::expand) fn expand_advice_ld(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_advice_load(instruction, 8, None, allocator)
}
