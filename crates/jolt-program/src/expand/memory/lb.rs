use super::*;

pub(in crate::expand) fn expand_lb(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_byte_load(instruction, allocator, true)
}
