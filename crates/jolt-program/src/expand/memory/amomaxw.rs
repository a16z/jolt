use super::*;

pub(in crate::expand) fn expand_amomaxw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, JoltInstructionKind::SLT, false, true)
}
