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
        sextw dividend.operand(), reg(rs1(instruction)?);
        sextw divisor.operand(), reg(rs2(instruction)?);
        advice quotient_magnitude.operand();
        negate_if dividend_magnitude.operand(), dividend.operand(), dividend.operand();
        negate_if divisor_magnitude.operand(), divisor.operand(), divisor.operand();
        assert_mul_u_no_overflow quotient_magnitude.operand(), divisor_magnitude.operand();
        mul remainder_magnitude.operand(), quotient_magnitude.operand(), divisor_magnitude.operand();
        assert_lte remainder_magnitude.operand(), dividend_magnitude.operand();
        sub remainder_magnitude.operand(), dividend_magnitude.operand(), remainder_magnitude.operand();
        assert_valid_unsigned_remainder remainder_magnitude.operand(), divisor_magnitude.operand();
        negate_if reg(rd(instruction)?), dividend.operand(), remainder_magnitude.operand();
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
