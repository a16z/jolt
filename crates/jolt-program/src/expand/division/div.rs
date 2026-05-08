use super::*;

pub(in crate::expand) fn expand_div(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, allocator, false, false)
}
