use super::*;

pub(in crate::expand) fn expand_sh(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(
        instruction,
        0xffff,
        Some(SourceInstructionKind::VirtualAssertHalfwordAlignment),
    )
}
