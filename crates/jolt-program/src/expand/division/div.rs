use super::*;
use crate::jolt_asm;

pub(in crate::expand) fn expand_div(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let dividend = reg(rs1(instruction)?);
    let divisor = reg(rs2(instruction)?);
    let quotient = asm.allocate()?;
    let dividend_magnitude = asm.allocate()?;
    let divisor_magnitude = asm.allocate()?;
    let quotient_sign = asm.allocate()?;
    let quotient_magnitude = asm.allocate()?;
    let remainder = asm.allocate()?;

    jolt_asm!(asm, {
        advice quotient.operand();
        assert_valid_div0 divisor, quotient.operand();
        negate_if dividend_magnitude.operand(), dividend, dividend;
        negate_if divisor_magnitude.operand(), divisor, divisor;
        xor quotient_sign.operand(), dividend, divisor;
        negate_if quotient_magnitude.operand(), quotient_sign.operand(), quotient.operand();
        assert_mul_u_no_overflow quotient_magnitude.operand(), divisor_magnitude.operand();
        mul remainder.operand(), quotient_magnitude.operand(), divisor_magnitude.operand();
        assert_lte remainder.operand(), dividend_magnitude.operand();
        sub remainder.operand(), dividend_magnitude.operand(), remainder.operand();
        assert_valid_unsigned_remainder remainder.operand(), divisor_magnitude.operand();
        addi reg(rd(instruction)?), quotient.operand(), 0;
    });

    asm.release_many([
        quotient,
        dividend_magnitude,
        divisor_magnitude,
        quotient_sign,
        quotient_magnitude,
        remainder,
    ]);
    asm.finalize()
}
