use super::*;

/// Lowers `AMOADD.W` through the shared word AMO template.
///
/// The helper extracts the old word, stores the low-word result of
/// `old + rs2` back into the containing doubleword, and returns old word
/// sign-extended in `rd`.
pub(in crate::expand) fn expand_amoaddw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, SourceInstructionKind::ADD)
}
