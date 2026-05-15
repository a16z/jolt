use super::*;

/// Lowers unsigned `AMOMINU.W` through the shared word min/max template.
///
/// The helper compares zero-extended old word and `rs2`, stores the unsigned
/// minimum into the selected word lane, and returns old word sign-extended.
pub(in crate::expand) fn expand_amominuw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLTU, true, false)
}
