use super::*;
use crate::jolt_asm;

pub(in crate::expand) fn expand_rem(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let dividend = reg(rs1(instruction)?);
    let divisor = reg(rs2(instruction)?);
    let quotient_magnitude = asm.allocate()?;
    let dividend_magnitude = asm.allocate()?;
    let divisor_magnitude = asm.allocate()?;
    let remainder_magnitude = asm.allocate()?;

    jolt_asm!(asm, {
        advice quotient_magnitude.operand();
        negate_if dividend_magnitude.operand(), dividend, dividend;
        negate_if divisor_magnitude.operand(), divisor, divisor;
        assert_mul_u_no_overflow quotient_magnitude.operand(), divisor_magnitude.operand();
        mul remainder_magnitude.operand(), quotient_magnitude.operand(), divisor_magnitude.operand();
        assert_lte remainder_magnitude.operand(), dividend_magnitude.operand();
        sub remainder_magnitude.operand(), dividend_magnitude.operand(), remainder_magnitude.operand();
        assert_valid_unsigned_remainder remainder_magnitude.operand(), divisor_magnitude.operand();
        negate_if reg(rd(instruction)?), dividend, remainder_magnitude.operand();
    });

    asm.release_many([
        quotient_magnitude,
        dividend_magnitude,
        divisor_magnitude,
        remainder_magnitude,
    ]);
    asm.finalize()
}
