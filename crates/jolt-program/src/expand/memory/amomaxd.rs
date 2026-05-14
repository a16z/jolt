use super::*;

/// Lowers signed `AMOMAX.D` through the shared doubleword min/max template.
///
/// The helper compares old memory and `rs2` with signed `SLT`, stores the
/// maximum, and returns the old memory value in `rd`.
pub(in crate::expand) fn expand_amomaxd(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_d(instruction, SourceInstructionKind::SLT, false)
}
