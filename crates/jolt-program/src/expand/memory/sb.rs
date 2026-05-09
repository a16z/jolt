use super::*;

pub(in crate::expand) fn expand_sb(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(instruction, 0xff, None)
}
