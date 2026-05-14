use super::*;

/// Lowers `SUBW` by subtracting at XLEN and then sign-extending the low word.
///
/// This mirrors the architectural RV64 word instruction: high bits from the
/// intermediate full-width subtraction are ignored, and bit 31 determines the
/// final sign extension.
pub(in crate::expand) fn expand_subw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_r(
        JoltInstructionKind::SUB,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        reg(rs2(instruction)?),
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );

    asm.finalize()
}
