use super::*;

pub(in crate::expand) fn expand_amoaddd(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, JoltInstructionKind::ADD)
}
