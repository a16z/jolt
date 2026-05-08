use super::*;

pub(in crate::expand) fn expand_amomaxw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_amo_minmax_w(
        instruction,
        JoltInstructionKind::SLT,
        false,
        true,
        allocator,
    )
}
