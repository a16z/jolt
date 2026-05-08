use super::*;

pub(in crate::expand) fn expand_signed_div_rem(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    word: bool,
    remainder_output: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let a0 = rs1(instruction)?;
    let a1 = rs2(instruction)?;
    let a2 = allocator.allocate()?;
    let a3 = allocator.allocate()?;
    let t0 = allocator.allocate()?;
    let t1 = allocator.allocate()?;
    let (mut t2, mut t3, t4) = if word {
        (
            allocator.allocate()?,
            allocator.allocate()?,
            Some(allocator.allocate()?),
        )
    } else {
        (0, 0, None)
    };
    let dividend = t4.unwrap_or(a0);
    let divisor = if word { t3 } else { a1 };
    let shmat = if word { 31 } else { 63 };
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.expand_j(JoltInstructionKind::VirtualAdvice, a2, 0)?;
    asm.expand_j(JoltInstructionKind::VirtualAdvice, a3, 0)?;
    if word {
        asm.expand_i(JoltInstructionKind::VirtualSignExtendWord, dividend, a0, 0)?;
        asm.expand_i(JoltInstructionKind::VirtualSignExtendWord, divisor, a1, 0)?;
    }
    asm.expand_b(JoltInstructionKind::VirtualAssertValidDiv0, divisor, a2, 0)?;
    asm.expand_r(
        if word {
            JoltInstructionKind::VirtualChangeDivisorW
        } else {
            JoltInstructionKind::VirtualChangeDivisor
        },
        t0,
        dividend,
        divisor,
    )?;

    if word {
        asm.expand_i(JoltInstructionKind::VirtualSignExtendWord, t1, a2, 0)?;
        asm.expand_b(JoltInstructionKind::VirtualAssertEQ, t1, a2, 0)?;
        asm.expand_i(JoltInstructionKind::SRAI, t2, a3, 32)?;
        asm.expand_b(JoltInstructionKind::VirtualAssertEQ, t2, 0, 0)?;
    } else {
        asm.expand_r(JoltInstructionKind::MULH, t1, a2, t0)?;
        t2 = asm.allocate()?;
        t3 = asm.allocate()?;
        asm.expand_r(JoltInstructionKind::MUL, t2, a2, t0)?;
        asm.expand_i(JoltInstructionKind::SRAI, t3, t2, shmat)?;
        asm.expand_b(JoltInstructionKind::VirtualAssertEQ, t1, t3, 0)?;
    }

    if word {
        asm.expand_i(JoltInstructionKind::SRAI, t2, dividend, shmat)?;
        asm.expand_r(JoltInstructionKind::XOR, t3, a3, t2)?;
        asm.expand_r(JoltInstructionKind::SUB, t3, t3, t2)?;
        asm.expand_r(JoltInstructionKind::MUL, t1, a2, t0)?;
        asm.expand_r(JoltInstructionKind::ADD, t1, t1, t3)?;
        asm.expand_b(JoltInstructionKind::VirtualAssertEQ, t1, dividend, 0)?;
        asm.expand_i(JoltInstructionKind::SRAI, t2, t0, 31)?;
        asm.expand_r(JoltInstructionKind::XOR, t1, t0, t2)?;
        asm.expand_r(JoltInstructionKind::SUB, t1, t1, t2)?;
        asm.expand_b(
            JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
            a3,
            t1,
            0,
        )?;
        asm.expand_i(
            JoltInstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            if remainder_output { t3 } else { a2 },
            0,
        )?;
    } else {
        let abs_divisor = if remainder_output { t2 } else { t3 };
        asm.expand_i(JoltInstructionKind::SRAI, t1, dividend, shmat)?;
        asm.expand_r(JoltInstructionKind::XOR, t3, a3, t1)?;
        asm.expand_r(JoltInstructionKind::SUB, t3, t3, t1)?;
        asm.expand_r(JoltInstructionKind::ADD, t2, t2, t3)?;
        asm.expand_b(JoltInstructionKind::VirtualAssertEQ, t2, a0, 0)?;
        asm.expand_i(JoltInstructionKind::SRAI, t1, t0, shmat)?;
        asm.expand_r(JoltInstructionKind::XOR, abs_divisor, t0, t1)?;
        asm.expand_r(JoltInstructionKind::SUB, abs_divisor, abs_divisor, t1)?;
        asm.expand_b(
            JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
            a3,
            abs_divisor,
            0,
        )?;
        asm.expand_i(
            JoltInstructionKind::ADDI,
            rd(instruction)?,
            if remainder_output { t3 } else { a2 },
            0,
        )?;
    }

    let mut released = vec![a2, a3, t0, t1, t2, t3];
    if let Some(t4) = t4 {
        released.push(t4);
    }
    for register in released {
        asm.release(register)?;
    }

    asm.finalize()
}

pub(in crate::expand) fn expand_unsigned_word_div_rem(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    remainder_output: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let rs1_extended = allocator.allocate()?;
    let rs2_extended = allocator.allocate()?;
    let quotient = allocator.allocate()?;
    let tmp = if remainder_output {
        quotient
    } else {
        allocator.allocate()?
    };

    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.expand_i(
        JoltInstructionKind::VirtualZeroExtendWord,
        rs1_extended,
        rs1(instruction)?,
        0,
    )?;
    asm.expand_i(
        JoltInstructionKind::VirtualZeroExtendWord,
        rs2_extended,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_j(JoltInstructionKind::VirtualAdvice, quotient, 0)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        quotient,
        rs2_extended,
        0,
    )?;
    asm.expand_r(JoltInstructionKind::MUL, tmp, quotient, rs2_extended)?;
    asm.expand_b(JoltInstructionKind::VirtualAssertLTE, tmp, rs1_extended, 0)?;
    asm.expand_r(JoltInstructionKind::SUB, tmp, rs1_extended, tmp)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        tmp,
        rs2_extended,
        0,
    )?;

    if remainder_output {
        asm.expand_i(
            JoltInstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            tmp,
            0,
        )?;
    } else {
        asm.expand_i(JoltInstructionKind::VirtualSignExtendWord, tmp, quotient, 0)?;
        asm.expand_b(
            JoltInstructionKind::VirtualAssertValidDiv0,
            rs2_extended,
            tmp,
            0,
        )?;
        asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, tmp, 0)?;
    }

    asm.release(rs1_extended)?;
    asm.release(rs2_extended)?;
    asm.release(quotient)?;
    if !remainder_output {
        asm.release(tmp)?;
    }

    asm.finalize()
}
