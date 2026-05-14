use super::*;

/// Lowers byte store `SB` through the shared masked doubleword update.
///
/// Byte stores have no alignment requirement; the helper updates only the
/// selected 8-bit lane of the containing doubleword.
pub(in crate::expand) fn expand_sb(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(instruction, 0xff, None)
}
