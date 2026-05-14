use super::*;

/// Lowers `MULW` by multiplying at XLEN and then imposing the RV64 word result.
///
/// RISC-V defines `MULW` as the low 32 bits of the product sign-extended to
/// 64 bits. The final virtual row is what discards any higher product bits.
pub(in crate::expand) fn expand_mulw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_r(
        JoltInstructionKind::MUL,
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
