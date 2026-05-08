use super::*;

pub(in crate::expand) fn expand_lbu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_byte_load(instruction, allocator, false)
}
