use super::*;

/// Lowers unsigned `AMOMINU.D` through the shared doubleword min/max template.
///
/// The helper compares old memory and `rs2` with unsigned `SLTU`, stores the
/// minimum, and returns the old memory value in `rd`.
pub(in crate::expand) fn expand_amominud(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_d(instruction, SourceInstructionKind::SLTU, true)
}
