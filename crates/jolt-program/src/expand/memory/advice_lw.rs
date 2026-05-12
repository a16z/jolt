use super::*;

pub(in crate::expand) fn expand_advice_lw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_advice_load(instruction, 4, Some(32), allocator)
}
