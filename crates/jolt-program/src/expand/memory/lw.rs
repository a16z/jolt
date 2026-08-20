use super::*;

/// Lowers signed word load `LW` by reading the containing aligned doubleword.
///
/// The sequence proves word alignment, loads the aligned 8-byte word, builds
/// the byte mask of the requested word lane from the effective address, and
/// extracts + sign-extends that lane into `rd` with a single fused lookup.
pub(in crate::expand) fn expand_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    // RAM is accessed at doubleword granularity here. The word alignment
    // assertion is still required by the source `LW` semantics; it also
    // guarantees the effective address's bits 0-1 are zero, which
    // `VirtualWindowMaskW` relies on (it reads only bit 2).
    asm.expand_address(
        SourceInstructionKind::VirtualAssertWordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    // v1 = containing doubleword address, v0 = effective (byte) address.
    asm.expand_i(
        SourceInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(SourceInstructionKind::LD, v1.operand(), v1.operand(), 0);
    // v0 = byte mask of the word lane at offset `ea mod 8`.
    asm.expand_i(
        SourceInstructionKind::VirtualWindowMaskW,
        v0.operand(),
        v0.operand(),
        0,
    );
    // rd = sign-extended word lane of the loaded doubleword.
    asm.expand_r(
        SourceInstructionKind::VirtualPextSigned,
        reg(rd(instruction)?),
        v1.operand(),
        v0.operand(),
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
