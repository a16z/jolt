use super::*;

/// Lowers `AMOAND.W` through the shared word AMO template.
///
/// The helper extracts the old word, stores `old & rs2` into the selected word
/// lane, and returns old word sign-extended in `rd`.
pub(in crate::expand) fn expand_amoandw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_w(instruction, SourceInstructionKind::AND)
}
