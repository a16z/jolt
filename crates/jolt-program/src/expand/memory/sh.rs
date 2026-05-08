use super::*;

pub(in crate::expand) fn expand_sh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    super::shared::expand_narrow_store(
        instruction,
        allocator,
        0xffff,
        Some(JoltInstructionKind::VirtualAssertHalfwordAlignment),
    )
}
