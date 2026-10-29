use super::shared::AccessWidth;
use super::*;

/// Lowers unsigned halfword load `LHU` through the shared floor-addressed load, which
/// extracts the addressed lane and zero-extends it.
pub(in crate::expand) fn expand_lhu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_load(instruction, AccessWidth::Halfword, false)
}
