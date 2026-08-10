use super::*;

/// Lowers unsigned word load `LWU` by extracting from the containing doubleword.
///
/// `VirtualWindowMaskW` builds the word lane's byte mask from the effective
/// address and `VirtualPext` extracts it, producing the zero-extended RV64
/// result.
pub(in crate::expand) fn expand_lwu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    // The source op still requires word alignment even though the physical
    // proof row reads the aligned containing doubleword.
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
    // v1 = containing doubleword address, v0 = byte offset within it.
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
    // rd = zero-extended word lane of the loaded doubleword.
    asm.expand_r(
        SourceInstructionKind::VirtualPext,
        reg(rd(instruction)?),
        v1.operand(),
        v0.operand(),
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
