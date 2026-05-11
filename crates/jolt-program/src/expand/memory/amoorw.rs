use super::*;

pub(in crate::expand) fn expand_amoorw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_w(instruction, InstructionKind::OR, allocator)
}
