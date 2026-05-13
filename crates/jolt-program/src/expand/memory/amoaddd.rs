use super::*;

pub(in crate::expand) fn expand_amoaddd(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, SourceInstructionKind::ADD)
}
