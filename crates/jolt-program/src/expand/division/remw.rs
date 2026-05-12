use super::*;

pub(in crate::expand) fn expand_remw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, allocator, true, true)
}
