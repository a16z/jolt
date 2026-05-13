use super::*;

pub(in crate::expand) fn expand_advice_ld(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 8)
}
