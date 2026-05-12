use super::*;

pub(in crate::expand) fn expand_amoorw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, JoltInstructionKind::OR)
}
