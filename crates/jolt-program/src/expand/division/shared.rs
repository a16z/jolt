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
        zextw dividend, reg(rs1(instruction)?);
        zextw divisor, reg(rs2(instruction)?);
        advice quotient;
        assert_mul_u_no_overflow quotient, divisor;
        mul remainder, quotient, divisor;
        assert_lte remainder, dividend;
        sub remainder, dividend, remainder;
        assert_valid_unsigned_remainder remainder, divisor;
    });

    if remainder_output {
        jolt_asm!(asm, {
            sextw reg(rd(instruction)?), remainder;
        });
    } else {
        jolt_asm!(asm, {
            sextw remainder, quotient;
            assert_valid_div0 divisor, remainder;
            addi reg(rd(instruction)?), remainder, 0;
        });
    }

    asm.release_many([dividend, divisor, quotient]);
    if !remainder_output {
        asm.release(remainder);
    }
    asm.finalize()
}
