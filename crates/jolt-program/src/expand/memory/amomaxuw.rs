use super::*;

/// Lowers unsigned `AMOMAXU.W` through the shared word min/max template.
///
/// The helper compares zero-extended old word and `rs2`, stores the unsigned
/// maximum into the selected word lane, and returns old word sign-extended.
pub(in crate::expand) fn expand_amomaxuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLTU, false, false)
}
