use super::*;

pub(in crate::expand) fn expand_advice_lb(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_advice_load(instruction, 1, Some(56), allocator)
}
