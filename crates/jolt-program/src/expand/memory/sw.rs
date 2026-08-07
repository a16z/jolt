use super::*;

/// Lowers word store `SW` through the shared masked doubleword update.
pub(in crate::expand) fn expand_sw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    super::shared::expand_narrow_store(
        instruction,
        SourceInstructionKind::VirtualLaneMaskW(jolt_riscv::instructions::VirtualLaneMaskW(())),
        Some(SourceInstructionKind::VirtualAssertWordAlignment),
    )
}
