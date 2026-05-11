use super::*;

pub(in crate::expand) fn expand_advice_lh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_advice_load(instruction, 2, Some(48), allocator)
}
