use super::*;

/// Lowers `AMOXOR.D` through the shared doubleword AMO template.
///
/// The shared helper loads the old doubleword, stores `old ^ rs2`, and returns
/// the old value in `rd`.
pub(in crate::expand) fn expand_amoxord(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_d(instruction, SourceInstructionKind::XOR)
}
