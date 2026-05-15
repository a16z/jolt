use super::*;

/// Lowers signed `AMOMIN.W` through the shared word min/max template.
///
/// The helper compares sign-extended old word and `rs2`, stores the signed
/// minimum into the selected word lane, and returns old word sign-extended.
pub(in crate::expand) fn expand_amominw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLT, true, true)
}
