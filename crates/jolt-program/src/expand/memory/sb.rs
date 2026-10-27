use super::shared::AccessWidth;
use super::*;

/// Lowers byte store `SB` through the shared floor-addressed
/// read-modify-write of the containing doubleword.
pub(in crate::expand) fn expand_sb(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(instruction, AccessWidth::Byte)
}
