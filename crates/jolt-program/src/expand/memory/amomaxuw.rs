use super::*;

pub(in crate::expand) fn expand_amomaxuw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, JoltInstructionKind::SLTU, false, false)
}
