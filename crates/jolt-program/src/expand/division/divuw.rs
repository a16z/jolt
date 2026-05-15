use super::*;

/// Lowers unsigned 32-bit `DIVUW` through the shared word-division verifier.
///
/// Both inputs are zero-extended to 32 bits, the quotient is proved as an
/// unsigned result, and the final quotient is sign-extended as required by RV64
/// word instructions.
pub(in crate::expand) fn expand_divuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_unsigned_word_div_rem(instruction, false)
}
