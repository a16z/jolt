use super::*;

pub(in crate::expand) fn expand_rem(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_signed_div_rem(instruction, false, true)
}
