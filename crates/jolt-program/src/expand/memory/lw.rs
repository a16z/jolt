use super::shared::AccessWidth;
use super::*;

/// Lowers signed word load `LW` through the shared floor-addressed load, which
/// extracts the addressed lane and sign-extends it.
pub(in crate::expand) fn expand_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_load(instruction, AccessWidth::Word, true)
}
