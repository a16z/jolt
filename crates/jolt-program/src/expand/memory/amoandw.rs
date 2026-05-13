use super::*;

pub(in crate::expand) fn expand_amoandw(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, JoltInstructionKind::AND)
}
