use super::*;

pub(in crate::expand) fn expand_amominuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLTU, true, false)
}
