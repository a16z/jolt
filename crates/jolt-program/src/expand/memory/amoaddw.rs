use super::*;

pub(in crate::expand) fn expand_amoaddw(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, SourceInstructionKind::ADD)
}
