use super::*;

pub(in crate::expand) fn expand_advice_lh(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 2, Some(48))
}
