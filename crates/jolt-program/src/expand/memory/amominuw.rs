use super::*;

pub(in crate::expand) fn expand_amominuw(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, JoltInstructionKind::SLTU, true, false)
}
