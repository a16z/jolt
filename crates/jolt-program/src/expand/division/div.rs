use super::*;

/// Lowers signed 64-bit `DIV` through the shared quotient/remainder verifier.
///
/// The shared recipe witnesses the signed quotient, proves its magnitude
/// against the operand magnitudes, and writes the quotient.
pub(in crate::expand) fn expand_div(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, false, false)
}
