use super::*;

pub(in crate::expand) fn expand_amoaddw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_w(instruction, InstructionKind::ADD, allocator)
}
