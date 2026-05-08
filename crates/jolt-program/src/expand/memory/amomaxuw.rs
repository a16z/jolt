use super::*;

pub(in crate::expand) fn expand_amomaxuw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, InstructionKind::SLTU, false, false, allocator)
}
