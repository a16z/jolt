use super::*;

pub(in crate::expand) fn expand_lhu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_halfword_load(instruction, false)
}
