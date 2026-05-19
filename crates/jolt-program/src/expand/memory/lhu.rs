use super::*;

/// Lowers unsigned halfword load `LHU` through the shared aligned extraction path.
///
/// The shared helper proves halfword alignment, extracts the selected
/// halfword, and zero-extends it to XLEN.
pub(in crate::expand) fn expand_lhu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_halfword_load(instruction, false)
}
