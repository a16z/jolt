use super::*;

/// Lowers unsigned word load `LWU` through the shared containing-doubleword load.
///
/// The helper builds the word lane's byte mask and extracts that lane into
/// `rd` zero-extended with a single fused lookup.
pub(in crate::expand) fn expand_lwu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_word_load(instruction, false)
}
