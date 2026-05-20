use super::*;

/// Lowers immediate `SLLI` to multiplication by the encoded power of two.
///
/// The source decoder has already normalized this as an RV64 instruction; the
/// immediate is masked to six bits and then emitted as a final `VirtualMULI`
/// row so the shift is represented as arithmetic in the proving circuit.
pub(in crate::expand) fn expand_slli(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        1i128 << shift,
    );

    asm.finalize()
}
