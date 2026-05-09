use super::*;

pub(in crate::expand) fn expand_amoxord(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, JoltInstructionKind::XOR)
}
