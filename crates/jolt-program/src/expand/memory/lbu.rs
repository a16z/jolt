use super::*;

pub(in crate::expand) fn expand_lbu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_byte_load(instruction, false)
}
