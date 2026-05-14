use super::*;

/// Lowers `ADDW` by emitting a full-width `ADD` followed by word sign
/// extension.
///
/// The full-width sum may contain arbitrary high bits. `VirtualSignExtendWord`
/// enforces the RV64 word-arithmetic contract that only the low 32-bit result
/// is kept and then sign-extended into the destination register.
pub(in crate::expand) fn expand_addw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_r(
        JoltInstructionKind::ADD,
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
