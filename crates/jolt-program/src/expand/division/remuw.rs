use super::*;

pub(in crate::expand) fn expand_remuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_unsigned_word_div_rem(instruction, true)
}
