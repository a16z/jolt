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
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    asm.emit_j(InstructionKind::VirtualAdvice, a2, 0)?;
    asm.emit_j(InstructionKind::VirtualAdvice, a3, 0)?;

    if word {
        asm.emit_i(InstructionKind::VirtualSignExtendWord, dividend, a0, 0)?;
        asm.emit_i(InstructionKind::VirtualSignExtendWord, divisor, a1, 0)?;
    }

    asm.emit_b(InstructionKind::VirtualAssertValidDiv0, divisor, a2, 0)?;
    asm.emit_r(
        if word {
            InstructionKind::VirtualChangeDivisorW
        } else {
            InstructionKind::VirtualChangeDivisor
        },
        t0,
        dividend,
        divisor,
    )?;

    if word {
        asm.emit_i(InstructionKind::VirtualSignExtendWord, t1, a2, 0)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t1, a2, 0)?;
        asm.emit_i(InstructionKind::SRAI, t2, a3, 32)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t2, 0, 0)?;
    } else {
        asm.emit_r(InstructionKind::MULH, t1, a2, t0)?;
        t2 = asm.allocator().allocate()?;
        t3 = asm.allocator().allocate()?;
        asm.emit_r(InstructionKind::MUL, t2, a2, t0)?;
        asm.emit_i(InstructionKind::SRAI, t3, t2, shmat)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t1, t3, 0)?;
    }

    if word {
        asm.emit_i(InstructionKind::SRAI, t2, dividend, shmat)?;
        asm.emit_r(InstructionKind::XOR, t3, a3, t2)?;
        asm.emit_r(InstructionKind::SUB, t3, t3, t2)?;
        asm.emit_r(InstructionKind::MUL, t1, a2, t0)?;
        asm.emit_r(InstructionKind::ADD, t1, t1, t3)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t1, dividend, 0)?;
        asm.emit_i(InstructionKind::SRAI, t2, t0, 31)?;
        asm.emit_r(InstructionKind::XOR, t1, t0, t2)?;
        asm.emit_r(InstructionKind::SUB, t1, t1, t2)?;
        asm.emit_b(
            InstructionKind::VirtualAssertValidUnsignedRemainder,
            a3,
            t1,
            0,
        )?;
        asm.emit_i(
            InstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            if remainder_output { t3 } else { a2 },
            0,
        )?;
    } else {
        asm.emit_i(InstructionKind::SRAI, t1, dividend, shmat)?;
        asm.emit_r(InstructionKind::XOR, t3, a3, t1)?;
        asm.emit_r(InstructionKind::SUB, t3, t3, t1)?;
        asm.emit_r(InstructionKind::ADD, t2, t2, t3)?;
        asm.emit_b(InstructionKind::VirtualAssertEQ, t2, a0, 0)?;
        asm.emit_i(InstructionKind::SRAI, t1, t0, shmat)?;
        asm.emit_r(
            InstructionKind::XOR,
            if remainder_output { t2 } else { t3 },
            t0,
            t1,
        )?;
        let abs_divisor = if remainder_output { t2 } else { t3 };
        asm.emit_r(InstructionKind::SUB, abs_divisor, abs_divisor, t1)?;
        asm.emit_b(
            InstructionKind::VirtualAssertValidUnsignedRemainder,
            a3,
            abs_divisor,
            0,
        )?;
        asm.emit_i(
            InstructionKind::ADDI,
            rd(instruction)?,
            if remainder_output { t3 } else { a2 },
            0,
        )?;
    }

    let sequence = asm.finalize()?;
    allocator.release(a2)?;
    allocator.release(a3)?;
    allocator.release(t0)?;
    allocator.release(t1)?;
    allocator.release(t2)?;
    allocator.release(t3)?;
    if let Some(t4) = t4 {
        allocator.release(t4)?;
    }
    Ok(sequence)
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
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualZeroExtendWord,
        rs1_extended,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(
        InstructionKind::VirtualZeroExtendWord,
        rs2_extended,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_j(InstructionKind::VirtualAdvice, quotient, 0)?;
    asm.emit_b(
        InstructionKind::VirtualAssertMulUNoOverflow,
        quotient,
        rs2_extended,
        0,
    )?;
    asm.emit_r(InstructionKind::MUL, tmp, quotient, rs2_extended)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, tmp, rs1_extended, 0)?;
    asm.emit_r(InstructionKind::SUB, tmp, rs1_extended, tmp)?;
    asm.emit_b(
        InstructionKind::VirtualAssertValidUnsignedRemainder,
        tmp,
        rs2_extended,
        0,
    )?;
    if remainder_output {
        asm.emit_i(
            InstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            tmp,
            0,
        )?;
    } else {
        asm.emit_i(InstructionKind::VirtualSignExtendWord, tmp, quotient, 0)?;
        asm.emit_b(
            InstructionKind::VirtualAssertValidDiv0,
            rs2_extended,
            tmp,
            0,
        )?;
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, tmp, 0)?;
    }
    let sequence = asm.finalize()?;
    allocator.release(rs1_extended)?;
    allocator.release(rs2_extended)?;
    allocator.release(quotient)?;
    if !remainder_output {
        allocator.release(tmp)?;
    }
    Ok(sequence)
}
