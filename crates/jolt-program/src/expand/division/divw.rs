use super::*;
use crate::jolt_asm;

pub(in crate::expand) fn expand_divw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let dividend = asm.allocate()?;
    let divisor = asm.allocate()?;
    let quotient = asm.allocate()?;
    let dividend_magnitude = asm.allocate()?;
    let divisor_magnitude = asm.allocate()?;
    let quotient_sign = asm.allocate()?;
    let quotient_magnitude = asm.allocate()?;
    let remainder = asm.allocate()?;

    jolt_asm!(asm, {
        sextw dividend, reg(rs1(instruction)?);
        sextw divisor, reg(rs2(instruction)?);
        advice quotient;
        assert_valid_div0 divisor, quotient;
        negate_if dividend_magnitude, dividend, dividend;
        negate_if divisor_magnitude, divisor, divisor;
        xor quotient_sign, dividend, divisor;
        negate_if quotient_magnitude, quotient_sign, quotient;
        assert_mul_u_no_overflow quotient_magnitude, divisor_magnitude;
        mul remainder, quotient_magnitude, divisor_magnitude;
        assert_lte remainder, dividend_magnitude;
        sub remainder, dividend_magnitude, remainder;
        assert_valid_unsigned_remainder remainder, divisor_magnitude;
        sextw reg(rd(instruction)?), quotient;
    });

    asm.release_many([
        dividend,
        divisor,
        quotient,
        dividend_magnitude,
        divisor_magnitude,
        quotient_sign,
        quotient_magnitude,
        remainder,
    ]);
    asm.finalize()
}
