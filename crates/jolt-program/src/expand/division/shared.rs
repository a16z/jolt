use super::*;

pub(in crate::expand) fn expand_signed_div_rem(
    instruction: &NormalizedInstruction,
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

    asm.dispatch_j(JoltInstructionKind::VirtualAdvice, a2.operand(), 0);
    asm.dispatch_j(JoltInstructionKind::VirtualAdvice, a3.operand(), 0);
    if word {
        asm.dispatch_i(JoltInstructionKind::VirtualSignExtendWord, dividend, a0, 0);
        asm.dispatch_i(JoltInstructionKind::VirtualSignExtendWord, divisor, a1, 0);
    }
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertValidDiv0,
        divisor,
        a2.operand(),
        0,
    );
    asm.dispatch_r(
        if word {
            JoltInstructionKind::VirtualChangeDivisorW
        } else {
            JoltInstructionKind::VirtualChangeDivisor
        },
        t0.operand(),
        dividend,
        divisor,
    );

    if word {
        let t2 = word_t2.ok_or(ExpansionError::MalformedInstruction("missing word temp"))?;
        asm.dispatch_i(
            JoltInstructionKind::VirtualSignExtendWord,
            t1.operand(),
            a2.operand(),
            0,
        );
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertEQ,
            t1.operand(),
            a2.operand(),
            0,
        );
        asm.dispatch_i(JoltInstructionKind::SRAI, t2.operand(), a3.operand(), 32);
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertEQ,
            t2.operand(),
            reg(0),
            0,
        );
    } else {
        asm.dispatch_r(
            JoltInstructionKind::MULH,
            t1.operand(),
            a2.operand(),
            t0.operand(),
        );
        let t2 = asm.allocate()?;
        let t3 = asm.allocate()?;
        asm.dispatch_r(
            JoltInstructionKind::MUL,
            t2.operand(),
            a2.operand(),
            t0.operand(),
        );
        asm.dispatch_i(JoltInstructionKind::SRAI, t3.operand(), t2.operand(), shmat);
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertEQ,
            t1.operand(),
            t3.operand(),
            0,
        );
        asm.release_many([t2, t3]);
    }

    if word {
        let t2 = word_t2.ok_or(ExpansionError::MalformedInstruction("missing word temp"))?;
        let t3 = word_t3.ok_or(ExpansionError::MalformedInstruction("missing word temp"))?;
        asm.dispatch_i(JoltInstructionKind::SRAI, t2.operand(), dividend, shmat);
        asm.dispatch_r(
            JoltInstructionKind::XOR,
            t3.operand(),
            a3.operand(),
            t2.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::SUB,
            t3.operand(),
            t3.operand(),
            t2.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::MUL,
            t1.operand(),
            a2.operand(),
            t0.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::ADD,
            t1.operand(),
            t1.operand(),
            t3.operand(),
        );
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertEQ,
            t1.operand(),
            dividend,
            0,
        );
        asm.dispatch_i(JoltInstructionKind::SRAI, t2.operand(), t0.operand(), 31);
        asm.dispatch_r(
            JoltInstructionKind::XOR,
            t1.operand(),
            t0.operand(),
            t2.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::SUB,
            t1.operand(),
            t1.operand(),
            t2.operand(),
        );
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
            a3.operand(),
            t1.operand(),
            0,
        );
        let output: RegisterOperand = if remainder_output {
            t3.operand()
        } else {
            a2.operand()
        };
        asm.dispatch_i(
            JoltInstructionKind::VirtualSignExtendWord,
            reg(rd(instruction)?),
            output,
            0,
        );
        asm.release_many([t2, t3]);
    } else {
        let t2 = asm.allocate()?;
        let t3 = asm.allocate()?;
        let abs_divisor = if remainder_output { t2 } else { t3 };
        asm.dispatch_i(JoltInstructionKind::SRAI, t1.operand(), dividend, shmat);
        asm.dispatch_r(
            JoltInstructionKind::XOR,
            t3.operand(),
            a3.operand(),
            t1.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::SUB,
            t3.operand(),
            t3.operand(),
            t1.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::ADD,
            t2.operand(),
            t2.operand(),
            t3.operand(),
        );
        asm.dispatch_b(JoltInstructionKind::VirtualAssertEQ, t2.operand(), a0, 0);
        asm.dispatch_i(JoltInstructionKind::SRAI, t1.operand(), t0.operand(), shmat);
        asm.dispatch_r(
            JoltInstructionKind::XOR,
            abs_divisor.operand(),
            t0.operand(),
            t1.operand(),
        );
        asm.dispatch_r(
            JoltInstructionKind::SUB,
            abs_divisor.operand(),
            abs_divisor.operand(),
            t1.operand(),
        );
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
            a3.operand(),
            abs_divisor.operand(),
            0,
        );
        let output: RegisterOperand = if remainder_output {
            t3.operand()
        } else {
            a2.operand()
        };
        asm.dispatch_i(JoltInstructionKind::ADDI, reg(rd(instruction)?), output, 0);
        asm.release_many([t2, t3]);
    }

    asm.release_many([a2, a3, t0, t1]);
    if let Some(t4) = t4 {
        asm.release(t4);
    }

    asm.finalize()
}

pub(in crate::expand) fn expand_unsigned_word_div_rem(
    instruction: &NormalizedInstruction,
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

    asm.dispatch_i(
        JoltInstructionKind::VirtualZeroExtendWord,
        rs1_extended.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.dispatch_i(
        JoltInstructionKind::VirtualZeroExtendWord,
        rs2_extended.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.dispatch_j(JoltInstructionKind::VirtualAdvice, quotient.operand(), 0);
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        quotient.operand(),
        rs2_extended.operand(),
        0,
    );
    asm.dispatch_r(
        JoltInstructionKind::MUL,
        tmp.operand(),
        quotient.operand(),
        rs2_extended.operand(),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertLTE,
        tmp.operand(),
        rs1_extended.operand(),
        0,
    );
    asm.dispatch_r(
        JoltInstructionKind::SUB,
        tmp.operand(),
        rs1_extended.operand(),
        tmp.operand(),
    );
    asm.dispatch_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        tmp.operand(),
        rs2_extended.operand(),
        0,
    );

    if remainder_output {
        asm.dispatch_i(
            JoltInstructionKind::VirtualSignExtendWord,
            reg(rd(instruction)?),
            tmp.operand(),
            0,
        );
    } else {
        asm.dispatch_i(
            JoltInstructionKind::VirtualSignExtendWord,
            tmp.operand(),
            quotient.operand(),
            0,
        );
        asm.dispatch_b(
            JoltInstructionKind::VirtualAssertValidDiv0,
            rs2_extended.operand(),
            tmp.operand(),
            0,
        );
        asm.dispatch_i(
            JoltInstructionKind::ADDI,
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
