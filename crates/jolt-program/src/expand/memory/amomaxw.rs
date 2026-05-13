use super::*;

pub(in crate::expand) fn expand_amomaxw(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLT, false, true)
}
