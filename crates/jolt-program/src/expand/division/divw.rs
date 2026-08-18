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
        sextw dividend.operand(), reg(rs1(instruction)?);
        sextw divisor.operand(), reg(rs2(instruction)?);
        advice quotient.operand();
        assert_valid_div0 divisor.operand(), quotient.operand();
        negate_if dividend_magnitude.operand(), dividend.operand(), dividend.operand();
        negate_if divisor_magnitude.operand(), divisor.operand(), divisor.operand();
        xor quotient_sign.operand(), dividend.operand(), divisor.operand();
        negate_if quotient_magnitude.operand(), quotient_sign.operand(), quotient.operand();
        assert_mul_u_no_overflow quotient_magnitude.operand(), divisor_magnitude.operand();
        mul remainder.operand(), quotient_magnitude.operand(), divisor_magnitude.operand();
        assert_lte remainder.operand(), dividend_magnitude.operand();
        sub remainder.operand(), dividend_magnitude.operand(), remainder.operand();
        assert_valid_unsigned_remainder remainder.operand(), divisor_magnitude.operand();
        sextw reg(rd(instruction)?), quotient.operand();
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
