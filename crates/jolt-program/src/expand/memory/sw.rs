use super::*;

pub(in crate::expand) fn expand_sw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(
        instruction,
        SourceInstructionKind::VirtualWindowMaskW,
        SourceInstructionKind::VirtualShiftDataW,
        Some(SourceInstructionKind::VirtualAssertWordAlignment),
    )
}
