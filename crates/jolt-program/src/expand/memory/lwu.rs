use super::*;

pub(in crate::expand) fn expand_lwu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_word_load(instruction, false)
}
