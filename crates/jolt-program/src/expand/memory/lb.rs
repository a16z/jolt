use super::shared::AccessWidth;
use super::*;

/// Lowers signed byte load `LB` through the shared floor-addressed load, which
/// extracts the addressed lane and sign-extends it.
pub(in crate::expand) fn expand_lb(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_load(instruction, AccessWidth::Byte, true)
}
