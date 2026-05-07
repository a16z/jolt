use super::*;

pub(in crate::expand) fn expand_divuw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_unsigned_word_div_rem(instruction, allocator, false)
}
