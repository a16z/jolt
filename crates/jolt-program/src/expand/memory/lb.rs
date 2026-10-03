use super::*;

/// Lowers signed byte load `LB` through the shared containing-doubleword load.
///
/// The helper extracts the selected byte and uses arithmetic shift-back to
/// sign-extend it to XLEN.
pub(in crate::expand) fn expand_lb(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_byte_load(instruction, true)
}
