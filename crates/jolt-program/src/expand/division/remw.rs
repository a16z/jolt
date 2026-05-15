use super::*;

/// Lowers signed 32-bit `REMW` through the shared signed division recipe.
///
/// The recipe works over sign-extended word operands and writes the proved
/// signed remainder, then applies the final RV64 word sign extension.
pub(in crate::expand) fn expand_remw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, true, true)
}
