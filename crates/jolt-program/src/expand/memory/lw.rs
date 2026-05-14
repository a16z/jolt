use super::*;

/// Lowers signed word load `LW` by reading the containing aligned doubleword.
///
/// The sequence proves word alignment, loads the aligned 8-byte word, shifts
/// the requested 32-bit lane down, and sign-extends that low word into `rd`.
pub(in crate::expand) fn expand_lw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    // RAM is accessed at doubleword granularity here. The word alignment
    // assertion is still required by the source `LW` semantics.
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
    asm.expand_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        SourceInstructionKind::SRL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        SourceInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
        reg(rd(instruction)?),
        v1.operand(),
        0,
    );
    asm.release(v0);
    asm.release(v1);

    asm.finalize()
}
