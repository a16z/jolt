use super::*;

pub(in crate::expand) fn expand_amomaxd(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_d(instruction, JoltInstructionKind::SLT, false)
}
