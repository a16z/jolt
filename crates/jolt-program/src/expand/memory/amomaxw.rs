use super::*;

/// Lowers signed `AMOMAX.W` through the shared word min/max template.
///
/// The helper compares sign-extended old word and `rs2`, stores the signed
/// maximum into the selected word lane, and returns old word sign-extended.
pub(in crate::expand) fn expand_amomaxw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_amo_minmax_w(instruction, SourceInstructionKind::SLT, false, true)
}
