use super::*;

/// Lowers `AdviceLD` to an eight-byte advice-tape read.
///
/// Full-width advice loads need no post-read sign extension because the advice
/// row already produces the complete XLEN value.
pub(in crate::expand) fn expand_advice_ld(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 8)
}
