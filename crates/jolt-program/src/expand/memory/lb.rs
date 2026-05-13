use super::*;

pub(in crate::expand) fn expand_lb(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_byte_load(instruction, true)
}
