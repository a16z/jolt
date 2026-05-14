use super::*;

/// Lowers signed 32-bit `DIVW` through the shared signed division recipe.
///
/// The recipe sign-extends both operands to their RV64 word values, proves the
/// quotient/remainder relation over those values, and sign-extends the quotient
/// as the final word result.
pub(in crate::expand) fn expand_divw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, true, false)
}
