use super::*;

pub(in crate::expand) fn expand_amominuw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_minmax_w(
        instruction,
        JoltInstructionKind::SLTU,
        true,
        false,
        allocator,
    )
}
