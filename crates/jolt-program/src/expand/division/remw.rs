use super::*;
use crate::jolt_asm;

pub(in crate::expand) fn expand_remw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let dividend = asm.allocate()?;
    let divisor = asm.allocate()?;
    let quotient_magnitude = asm.allocate()?;
    let dividend_magnitude = asm.allocate()?;
    let divisor_magnitude = asm.allocate()?;
    let remainder_magnitude = asm.allocate()?;

    jolt_asm!(asm, {
        sextw dividend, reg(rs1(instruction)?);
        sextw divisor, reg(rs2(instruction)?);
        advice quotient_magnitude;
        negate_if dividend_magnitude, dividend, dividend;
        negate_if divisor_magnitude, divisor, divisor;
        assert_mul_u_no_overflow quotient_magnitude, divisor_magnitude;
        mul remainder_magnitude, quotient_magnitude, divisor_magnitude;
        assert_lte remainder_magnitude, dividend_magnitude;
        sub remainder_magnitude, dividend_magnitude, remainder_magnitude;
        assert_valid_unsigned_remainder remainder_magnitude, divisor_magnitude;
        negate_if reg(rd(instruction)?), dividend, remainder_magnitude;
    });

    asm.release_many([
        dividend,
        divisor,
        quotient_magnitude,
        dividend_magnitude,
        divisor_magnitude,
        remainder_magnitude,
    ]);
    asm.finalize()
}
