use super::*;

pub(in crate::expand) fn expand_amoxord(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, SourceInstructionKind::XOR)
}
