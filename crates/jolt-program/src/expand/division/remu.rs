use super::*;

/// Lowers unsigned 64-bit `REMU` by witnessing the quotient and deriving the
/// remainder.
///
/// The quotient is never exposed. The sequence proves multiplication does not
/// overflow, subtracts `q * divisor` from the dividend, and checks the result
/// satisfies the unsigned remainder bound, with divisor zero admitted by the
/// virtual assertion.
pub(in crate::expand) fn expand_remu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;

    // v0 starts as the quotient witness and is then reused for q * divisor and
    // finally for the derived remainder.
    asm.expand_j(SourceInstructionKind::VirtualAdvice, v0.operand(), 0);
    asm.expand_b(
        SourceInstructionKind::VirtualAssertMulUNoOverflow,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v0.operand(),
        v0.operand(),
        reg(rs2(instruction)?),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::SUB,
        v0.operand(),
        reg(rs1(instruction)?),
        v0.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release(v0);

    asm.finalize()
}
