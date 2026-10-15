use super::*;

/// Lowers word store `SW` through the shared masked doubleword update.
///
/// The shared helper proves word alignment, updates only the selected 32-bit
/// lane, and writes the containing doubleword back.
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
