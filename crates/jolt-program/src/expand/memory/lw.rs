use super::*;

/// Lowers signed word load `LW` through the shared containing-doubleword load.
///
/// The helper builds the word lane's byte mask and extracts + sign-extends
/// that lane into `rd` with a single fused lookup.
pub(in crate::expand) fn expand_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_word_load(instruction, true)
}
