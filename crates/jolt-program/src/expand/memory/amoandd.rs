use super::*;

pub(in crate::expand) fn expand_amoandd(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, JoltInstructionKind::AND)
}
