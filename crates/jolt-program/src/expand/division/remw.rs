use super::*;

pub(in crate::expand) fn expand_remw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, true, true)
}
