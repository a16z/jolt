use super::*;

pub(in crate::expand) fn expand_divuw(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_unsigned_word_div_rem(instruction, false)
}
