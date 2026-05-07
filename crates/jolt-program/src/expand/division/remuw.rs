use super::*;

pub(in crate::expand) fn expand_remuw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_unsigned_word_div_rem(instruction, allocator, true)
}
