use super::*;

/// Lowers `ADDIW` by doing the addition in the final row universe and then
/// forcing the architectural RV64 word result.
///
/// RISC-V word arithmetic keeps only the low 32 bits and sign-extends bit 31
/// back to XLEN. The final `VirtualSignExtendWord` row is therefore part of
/// the source instruction's semantics, not a cleanup step.
pub(in crate::expand) fn expand_addiw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );

    asm.finalize()
}
