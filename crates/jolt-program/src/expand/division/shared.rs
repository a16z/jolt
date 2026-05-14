use super::*;

/// Builds the signed `DIV`/`REM` and `DIVW`/`REMW` verifier sequence.
///
/// The tracer supplies quotient `a2` and remainder data `a3` as advice. This
/// recipe proves the RISC-V signed-division contract around those witnesses:
/// divide-by-zero quotient, signed overflow (`MIN / -1`), product/remainder
/// recomposition, and `abs(remainder) < abs(divisor)` whenever the divisor is
/// nonzero. `word` selects 32-bit operand normalization and final word
/// sign-extension; `remainder_output` selects which proved value is copied to
/// `rd`.
pub(in crate::expand) fn expand_signed_div_rem(
    instruction: &SourceInstructionRow,
    word: bool,
    remainder_output: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let a0 = reg(rs1(instruction)?);
    let a1 = reg(rs2(instruction)?);
    let a2 = asm.allocate()?;
    let a3 = asm.allocate()?;
    let t0 = asm.allocate()?;
    let t1 = asm.allocate()?;
    let (word_t2, word_t3) = if word {
        (Some(asm.allocate()?), Some(asm.allocate()?))
    } else {
        (None, None)
    };
    let t4 = if word { Some(asm.allocate()?) } else { None };
    let dividend: RegisterOperand = t4.map_or(a0, TempId::operand);
    let divisor: RegisterOperand = word_t3.map_or(a1, TempId::operand);
    let shmat = if word { 31 } else { 63 };

    // a2 is the quotient witness. a3 participates in the remainder proof and
    // later becomes the signed remainder candidate for REM/REMW.
    asm.expand_j(SourceInstructionKind::VirtualAdvice, a2.operand(), 0);
    asm.expand_j(SourceInstructionKind::VirtualAdvice, a3.operand(), 0);
    if word {
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord,
            dividend,
            a0,
            0,
        );
        asm.expand_i(SourceInstructionKind::VirtualSignExtendWord, divisor, a1, 0);
    }
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidDiv0,
        divisor,
        a2.operand(),
        0,
    );
    // RISC-V signed overflow (`MIN / -1`) returns the dividend as the
    // quotient and zero remainder. Replacing the divisor by 1 in that one case
    // lets the same product/remainder proof cover normal and overflow inputs.
    asm.expand_r(
        if word {
            SourceInstructionKind::VirtualChangeDivisorW
        } else {
            SourceInstructionKind::VirtualChangeDivisor
        },
        t0.operand(),
        dividend,
        divisor,
    );

    if word {
        // Word mode proves that the quotient is already sign-extended from 32
        // bits and that the supplied remainder data fits in the low word.
        let t2 = word_t2.ok_or(ExpansionError::MalformedInstruction("missing word temp"))?;
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord,
            t1.operand(),
            a2.operand(),
            0,
        );
        asm.expand_b(
            SourceInstructionKind::VirtualAssertEQ,
            t1.operand(),
            a2.operand(),
            0,
        );
        asm.expand_i(SourceInstructionKind::SRAI, t2.operand(), a3.operand(), 32);
        asm.expand_b(
            SourceInstructionKind::VirtualAssertEQ,
            t2.operand(),
            reg(0),
            0,
        );
    } else {
        asm.expand_r(
            SourceInstructionKind::MULH,
            t1.operand(),
            a2.operand(),
            t0.operand(),
        );
        let t2 = asm.allocate()?;
        let t3 = asm.allocate()?;
        asm.expand_r(
            SourceInstructionKind::MUL,
            t2.operand(),
            a2.operand(),
            t0.operand(),
        );
        asm.expand_i(
            SourceInstructionKind::SRAI,
            t3.operand(),
            t2.operand(),
            shmat,
        );
        asm.expand_b(
            SourceInstructionKind::VirtualAssertEQ,
            t1.operand(),
            t3.operand(),
            0,
        );
        asm.release_many([t2, t3]);
    }

    if word {
        // Recompose the signed dividend from quotient, adjusted divisor, and
        // signed remainder, then bound the absolute remainder by the absolute
        // adjusted divisor.
        let t2 = word_t2.ok_or(ExpansionError::MalformedInstruction("missing word temp"))?;
        let t3 = word_t3.ok_or(ExpansionError::MalformedInstruction("missing word temp"))?;
        asm.expand_i(SourceInstructionKind::SRAI, t2.operand(), dividend, shmat);
        asm.expand_r(
            SourceInstructionKind::XOR,
            t3.operand(),
            a3.operand(),
            t2.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::SUB,
            t3.operand(),
            t3.operand(),
            t2.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::MUL,
            t1.operand(),
            a2.operand(),
            t0.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::ADD,
            t1.operand(),
            t1.operand(),
            t3.operand(),
        );
        asm.expand_b(
            SourceInstructionKind::VirtualAssertEQ,
            t1.operand(),
            dividend,
            0,
        );
        asm.expand_i(SourceInstructionKind::SRAI, t2.operand(), t0.operand(), 31);
        asm.expand_r(
            SourceInstructionKind::XOR,
            t1.operand(),
            t0.operand(),
            t2.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::SUB,
            t1.operand(),
            t1.operand(),
            t2.operand(),
        );
        asm.expand_b(
            SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
            a3.operand(),
            t1.operand(),
            0,
        );
        let output: RegisterOperand = if remainder_output {
            t3.operand()
        } else {
            a2.operand()
        };
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord,
            reg(rd(instruction)?),
            output,
            0,
        );
        asm.release_many([t2, t3]);
    } else {
        // 64-bit signed division follows the same proof shape as word mode,
        // but computes temporary absolute values for the remainder and adjusted
        // divisor directly in 64-bit space.
        let t2 = asm.allocate()?;
        let t3 = asm.allocate()?;
        let abs_divisor = if remainder_output { t2 } else { t3 };
        asm.expand_i(SourceInstructionKind::SRAI, t1.operand(), dividend, shmat);
        asm.expand_r(
            SourceInstructionKind::XOR,
            t3.operand(),
            a3.operand(),
            t1.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::SUB,
            t3.operand(),
            t3.operand(),
            t1.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::ADD,
            t2.operand(),
            t2.operand(),
            t3.operand(),
        );
        asm.expand_b(SourceInstructionKind::VirtualAssertEQ, t2.operand(), a0, 0);
        asm.expand_i(
            SourceInstructionKind::SRAI,
            t1.operand(),
            t0.operand(),
            shmat,
        );
        asm.expand_r(
            SourceInstructionKind::XOR,
            abs_divisor.operand(),
            t0.operand(),
            t1.operand(),
        );
        asm.expand_r(
            SourceInstructionKind::SUB,
            abs_divisor.operand(),
            abs_divisor.operand(),
            t1.operand(),
        );
        asm.expand_b(
            SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
            a3.operand(),
            abs_divisor.operand(),
            0,
        );
        let output: RegisterOperand = if remainder_output {
            t3.operand()
        } else {
            a2.operand()
        };
        asm.expand_i(
            SourceInstructionKind::ADDI,
            reg(rd(instruction)?),
            output,
            0,
        );
        asm.release_many([t2, t3]);
    }

    asm.release_many([a2, a3, t0, t1]);
    if let Some(t4) = t4 {
        asm.release(t4);
    }

    asm.finalize()
}

/// Builds the unsigned `DIVUW`/`REMUW` verifier sequence.
///
/// Word unsigned division first zero-extends both source operands. A quotient
/// witness is then constrained by
/// `dividend = quotient * divisor + remainder` with `remainder < divisor`,
/// except that the RISC-V divisor-zero quotient is admitted by
/// `VirtualAssertValidDiv0`. The chosen output is sign-extended as a 32-bit
/// RV64 word result.
pub(in crate::expand) fn expand_unsigned_word_div_rem(
    instruction: &SourceInstructionRow,
    remainder_output: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let rs1_extended = asm.allocate()?;
    let rs2_extended = asm.allocate()?;
    let quotient = asm.allocate()?;
    let tmp = if remainder_output {
        quotient
    } else {
        asm.allocate()?
    };

    // The relation is unsigned 32-bit, so both architectural inputs are
    // explicitly normalized before any quotient checks are emitted.
    asm.expand_i(
        SourceInstructionKind::VirtualZeroExtendWord,
        rs1_extended.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::VirtualZeroExtendWord,
        rs2_extended.operand(),
        reg(rs2(instruction)?),
        0,
    );
    // quotient is advice; tmp is first q * divisor and then the derived
    // remainder unless this is the quotient-output path.
    asm.expand_j(SourceInstructionKind::VirtualAdvice, quotient.operand(), 0);
    asm.expand_b(
        SourceInstructionKind::VirtualAssertMulUNoOverflow,
        quotient.operand(),
        rs2_extended.operand(),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        tmp.operand(),
        quotient.operand(),
        rs2_extended.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        tmp.operand(),
        rs1_extended.operand(),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::SUB,
        tmp.operand(),
        rs1_extended.operand(),
        tmp.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
        tmp.operand(),
        rs2_extended.operand(),
        0,
    );

    if remainder_output {
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord,
            reg(rd(instruction)?),
            tmp.operand(),
            0,
        );
    } else {
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord,
            tmp.operand(),
            quotient.operand(),
            0,
        );
        asm.expand_b(
            SourceInstructionKind::VirtualAssertValidDiv0,
            rs2_extended.operand(),
            tmp.operand(),
            0,
        );
        asm.expand_i(
            SourceInstructionKind::ADDI,
            reg(rd(instruction)?),
            tmp.operand(),
            0,
        );
    }

    asm.release_many([rs1_extended, rs2_extended, quotient]);
    if !remainder_output {
        asm.release(tmp);
    }

    asm.finalize()
}
