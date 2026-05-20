use super::*;

/// Lowers `AdviceLH` to a two-byte advice-tape read and sign-extension.
///
/// Advice loads are not RAM reads; the byte length is encoded in the virtual
/// row and the shared helper performs the architectural narrow-load extension.
pub(in crate::expand) fn expand_advice_lh(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 2)
}
