use super::*;

pub(in crate::expand) fn expand_advice_lb(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 1, Some(56))
}
