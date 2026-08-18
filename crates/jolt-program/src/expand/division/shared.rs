use super::*;
use crate::jolt_asm;

pub(in crate::expand) fn expand_unsigned_word_div_rem(
    instruction: &SourceInstructionRow,
    remainder_output: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let dividend = asm.allocate()?;
    let divisor = asm.allocate()?;
    let quotient = asm.allocate()?;
    let remainder = if remainder_output {
        quotient
    } else {
        asm.allocate()?
    };

    jolt_asm!(asm, {
        zextw dividend.operand(), reg(rs1(instruction)?);
        zextw divisor.operand(), reg(rs2(instruction)?);
        advice quotient.operand();
        assert_mul_u_no_overflow quotient.operand(), divisor.operand();
        mul remainder.operand(), quotient.operand(), divisor.operand();
        assert_lte remainder.operand(), dividend.operand();
        sub remainder.operand(), dividend.operand(), remainder.operand();
        assert_valid_unsigned_remainder remainder.operand(), divisor.operand();
    });

    if remainder_output {
        jolt_asm!(asm, {
            sextw reg(rd(instruction)?), remainder.operand();
        });
    } else {
        jolt_asm!(asm, {
            sextw remainder.operand(), quotient.operand();
            assert_valid_div0 divisor.operand(), remainder.operand();
            addi reg(rd(instruction)?), remainder.operand(), 0;
        });
    }

    asm.release_many([dividend, divisor, quotient]);
    if !remainder_output {
        asm.release(remainder);
    }
    asm.finalize()
}
