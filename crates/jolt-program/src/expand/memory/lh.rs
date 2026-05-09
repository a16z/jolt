use super::*;

pub(in crate::expand) fn expand_lh(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_halfword_load(instruction, true)
}
