use super::*;

pub(in crate::expand) fn expand_sh(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(
        instruction,
        SourceInstructionKind::VirtualWindowMaskH,
        SourceInstructionKind::VirtualShiftDataH,
        Some(SourceInstructionKind::VirtualAssertHalfwordAlignment),
    )
}
