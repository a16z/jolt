use super::*;

/// Lowers signed 64-bit `REM` through the shared quotient/remainder verifier.
///
/// The same relation that proves `DIV` also proves the remainder; this wrapper
/// selects the remainder path and copies that value to `rd`.
pub(in crate::expand) fn expand_rem(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, false, true)
}
