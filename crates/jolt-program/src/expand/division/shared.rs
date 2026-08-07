use super::*;

/// Builds the signed `DIV`/`REM` and `DIVW`/`REMW` verifier sequence.
///
/// The tracer supplies either the signed quotient or its magnitude as advice.
/// The shared body proves unsigned division over operand magnitudes, then
/// applies the RISC-V signs to the selected result.
pub(in crate::expand) fn expand_signed_div_rem(
    instruction: &SourceInstructionRow,
    word: bool,
    remainder_output: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let rs1 = reg(rs1(instruction)?);
    let rs2 = reg(rs2(instruction)?);
    let word_dividend = if word { Some(asm.allocate()?) } else { None };
    let word_divisor = if word { Some(asm.allocate()?) } else { None };
    let dividend = word_dividend.map_or(rs1, TempId::operand);
    let divisor = word_divisor.map_or(rs2, TempId::operand);
    let quotient = asm.allocate()?;
    let sign_source = asm.allocate()?;
    let abs_dividend = asm.allocate()?;
    let quotient_magnitude = asm.allocate()?;
    let abs_divisor = asm.allocate()?;
    let product = asm.allocate()?;

    if word {
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord(
                jolt_riscv::instructions::VirtualSignExtendWord(()),
            ),
            dividend,
            rs1,
            0,
        );
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord(
                jolt_riscv::instructions::VirtualSignExtendWord(()),
            ),
            divisor,
            rs2,
            0,
        );
    }

    asm.expand_j(
        SourceInstructionKind::VirtualAdvice(jolt_riscv::instructions::VirtualAdvice(())),
        quotient.operand(),
        0,
    );
    if !remainder_output {
        asm.expand_b(
            SourceInstructionKind::VirtualAssertValidDiv0,
            divisor,
            quotient.operand(),
            0,
        );
    }

    asm.emit_r(
        JoltInstructionKind::VirtualNegateIf(jolt_riscv::instructions::VirtualNegateIf(())),
        abs_dividend.operand(),
        dividend,
        dividend,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualNegateIf(jolt_riscv::instructions::VirtualNegateIf(())),
        abs_divisor.operand(),
        divisor,
        divisor,
    );

    let proved_quotient = if remainder_output {
        quotient.operand()
    } else {
        asm.expand_r(
            SourceInstructionKind::XOR,
            sign_source.operand(),
            dividend,
            divisor,
        );
        asm.emit_r(
            JoltInstructionKind::VirtualNegateIf(jolt_riscv::instructions::VirtualNegateIf(())),
            quotient_magnitude.operand(),
            sign_source.operand(),
            quotient.operand(),
        );
        quotient_magnitude.operand()
    };

    asm.expand_b(
        SourceInstructionKind::VirtualAssertMulUNoOverflow,
        proved_quotient,
        abs_divisor.operand(),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        product.operand(),
        proved_quotient,
        abs_divisor.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        product.operand(),
        abs_dividend.operand(),
        0,
    );
    asm.expand_r(
        SourceInstructionKind::SUB,
        product.operand(),
        abs_dividend.operand(),
        product.operand(),
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertValidUnsignedRemainder,
        product.operand(),
        abs_divisor.operand(),
        0,
    );

    if remainder_output {
        asm.emit_r(
            JoltInstructionKind::VirtualNegateIf(jolt_riscv::instructions::VirtualNegateIf(())),
            reg(rd(instruction)?),
            dividend,
            product.operand(),
        );
    } else if word {
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord(
                jolt_riscv::instructions::VirtualSignExtendWord(()),
            ),
            reg(rd(instruction)?),
            quotient.operand(),
            0,
        );
    } else {
        asm.expand_i(
            SourceInstructionKind::ADDI,
            reg(rd(instruction)?),
            quotient.operand(),
            0,
        );
    }

    asm.release_many([
        quotient,
        sign_source,
        abs_dividend,
        quotient_magnitude,
        abs_divisor,
        product,
    ]);
    if let Some(word_dividend) = word_dividend {
        asm.release(word_dividend);
    }
    if let Some(word_divisor) = word_divisor {
        asm.release(word_divisor);
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
        SourceInstructionKind::VirtualZeroExtendWord(
            jolt_riscv::instructions::VirtualZeroExtendWord(()),
        ),
        rs1_extended.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::VirtualZeroExtendWord(
            jolt_riscv::instructions::VirtualZeroExtendWord(()),
        ),
        rs2_extended.operand(),
        reg(rs2(instruction)?),
        0,
    );
    // quotient is advice; tmp is first q * divisor and then the derived
    // remainder unless this is the quotient-output path.
    asm.expand_j(
        SourceInstructionKind::VirtualAdvice(jolt_riscv::instructions::VirtualAdvice(())),
        quotient.operand(),
        0,
    );
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
            SourceInstructionKind::VirtualSignExtendWord(
                jolt_riscv::instructions::VirtualSignExtendWord(()),
            ),
            reg(rd(instruction)?),
            tmp.operand(),
            0,
        );
    } else {
        asm.expand_i(
            SourceInstructionKind::VirtualSignExtendWord(
                jolt_riscv::instructions::VirtualSignExtendWord(()),
            ),
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
