use super::*;

pub(in crate::expand) fn expand_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_word_load(instruction, true)
}
