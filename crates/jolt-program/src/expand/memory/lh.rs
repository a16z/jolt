use super::*;

/// Lowers signed halfword load `LH` through the shared aligned extraction path.
///
/// The shared helper proves halfword alignment, extracts the selected
/// halfword from the containing doubleword, and sign-extends it.
pub(in crate::expand) fn expand_lh(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_halfword_load(instruction, true)
}
