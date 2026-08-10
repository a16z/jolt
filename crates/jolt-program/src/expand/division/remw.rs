use super::*;

/// Lowers signed 32-bit `REMW` through the shared signed division recipe.
///
/// The recipe proves the remainder magnitude over sign-extended word operands,
/// then reapplies the dividend sign directly into `rd`.
pub(in crate::expand) fn expand_remw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, true, true)
}
