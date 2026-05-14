use super::*;

/// Lowers unsigned 64-bit `DIVU` by witnessing a quotient and proving it is
/// the unique valid quotient for `(rs1, rs2)`.
///
/// The sequence accepts the RISC-V divide-by-zero result through
/// `VirtualAssertValidDiv0`. Otherwise it proves `q * divisor` does not
/// overflow, `q * divisor <= dividend`, and the derived remainder is either
/// below the divisor or the divisor is zero.
pub(in crate::expand) fn expand_divu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    // v0 is the quotient supplied by the tracer. The following assertions bind
    // it to the architectural unsigned division relation before copying to rd.
    asm.expand_j(SourceInstructionKind::VirtualAdvice, v0.operand(), 0);
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidDiv0,
        reg(rs2(instruction)?),
        v0.operand(),
        0,
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertMulUNoOverflow,
        v0.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v1.operand(),
        v0.operand(),
        reg(rs2(instruction)?),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        v1.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::SUB,
        v1.operand(),
        reg(rs1(instruction)?),
        v1.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
        v1.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}
