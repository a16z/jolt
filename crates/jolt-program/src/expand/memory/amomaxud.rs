use super::*;

pub(in crate::expand) fn expand_amomaxud(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_d(instruction, SourceInstructionKind::SLTU, false)
}
