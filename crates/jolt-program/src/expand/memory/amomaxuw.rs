use super::*;

pub(in crate::expand) fn expand_amomaxuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLTU, false, false)
}
