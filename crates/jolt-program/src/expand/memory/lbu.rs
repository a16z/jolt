use super::*;

/// Lowers unsigned byte load `LBU` through the shared containing-doubleword load.
///
/// The helper extracts the selected byte and uses logical shift-back so the
/// upper bits are zero-filled.
pub(in crate::expand) fn expand_lbu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_byte_load(instruction, false)
}
