use super::*;

/// Lowers `AdviceLW` to a four-byte advice-tape read and word sign-extension.
///
/// The shared helper keeps advice traffic separate from RAM while preserving
/// the same sign-extension behavior as a 32-bit signed load.
pub(in crate::expand) fn expand_advice_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_advice_load(instruction, 4)
}
