use super::*;

pub(in crate::expand) fn expand_amomind(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_d(instruction, JoltInstructionKind::SLT, true)
}
