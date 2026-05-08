use super::*;

pub(in crate::expand) fn expand_amominw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, InstructionKind::SLT, true, true, allocator)
}
