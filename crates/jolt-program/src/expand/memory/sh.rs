use super::*;

/// Lowers halfword store `SH` through the shared masked doubleword update.
///
/// The shared helper proves halfword alignment, updates only the selected
/// 16-bit lane, and writes the containing doubleword back.
pub(in crate::expand) fn expand_sh(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(
        instruction,
        0xffff,
        Some(SourceInstructionKind::VirtualAssertHalfwordAlignment),
    )
}
