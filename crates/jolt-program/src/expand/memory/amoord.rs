use super::*;

pub(in crate::expand) fn expand_amoord(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, SourceInstructionKind::OR)
}
