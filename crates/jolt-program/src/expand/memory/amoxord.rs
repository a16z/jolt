use super::*;

pub(in crate::expand) fn expand_amoxord(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_d(instruction, JoltInstructionKind::XOR, allocator)
}
