use super::*;

/// Lowers unsigned word load `LWU` by extracting from the containing doubleword.
///
/// The loaded 32-bit lane is moved into the high half and logically shifted
/// back down, producing the zero-extended RV64 result.
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
    // XOR with 4 selects the opposite 32-bit lane after the doubleword is
    // shifted left, so the requested word lands in bits 63:32.
    asm.expand_i(SourceInstructionKind::XORI, v0.operand(), v0.operand(), 4);
    asm.expand_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        SourceInstructionKind::SLL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        SourceInstructionKind::SRLI,
        reg(rd(instruction)?),
        v1.operand(),
        32,
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
