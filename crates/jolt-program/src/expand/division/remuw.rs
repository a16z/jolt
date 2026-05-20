use super::*;

/// Lowers unsigned 32-bit `REMUW` through the shared word-division verifier.
///
/// The recipe proves the unsigned 32-bit quotient/remainder relation and
/// sign-extends the derived remainder to match RV64 word-result semantics.
pub(in crate::expand) fn expand_remuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_unsigned_word_div_rem(instruction, true)
}
