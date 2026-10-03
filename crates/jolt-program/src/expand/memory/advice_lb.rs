use super::*;

/// Lowers `AdviceLB` to an advice-tape byte read and sign-extension sequence.
///
/// The shared helper reads exactly one byte from the advice tape and then
/// sign-extends that byte to the architectural register width.
pub(in crate::expand) fn expand_advice_lb(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 1)
}
