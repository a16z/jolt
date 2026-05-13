use super::*;

pub(in crate::expand) fn expand_amoaddw(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, SourceInstructionKind::ADD)
}
