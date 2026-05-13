use super::*;

pub(in crate::expand) fn expand_amoorw(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, SourceInstructionKind::OR)
}
