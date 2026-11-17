use super::*;
use jolt_riscv::SourceInstructionKind as Kind;

/// Loads the containing doubleword, then extracts the addressed unsigned byte.
/// The interleaved extraction range-checks the entire RAM word, including when
/// the architectural destination is x0 (rewritten by the expansion builder).
pub(in crate::expand) fn expand_lbu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let doubleword = asm.allocate()?;
    let base = reg(rs1(instruction)?);
    let offset = instruction.operands.imm;
    asm.emit_ld(Kind::VirtualLoadB, doubleword.operand(), base, offset);
    asm.emit_shift_data(
        Kind::VirtualExtractByteUnsigned,
        reg(rd(instruction)?),
        doubleword.operand(),
        base,
        offset,
    );
    asm.release(doubleword);
    asm.finalize()
}
