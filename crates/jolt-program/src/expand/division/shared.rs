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
    let mut state = core::ExpansionState::new(allocator);
    let mut sequence = core::ExpansionSequence::new(instruction);

    let mut ops = vec![
        grammar::ExpansionOp::Expand(grammar::RowTemplate::j(
            JoltInstructionKind::VirtualAdvice,
            a2,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::j(
            JoltInstructionKind::VirtualAdvice,
            a3,
            0,
        )),
    ];
    if word {
        ops.extend([
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualSignExtendWord,
                dividend,
                a0,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualSignExtendWord,
                divisor,
                a1,
                0,
            )),
        ]);
    }
    ops.extend([
        grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
            JoltInstructionKind::VirtualAssertValidDiv0,
            divisor,
            a2,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            if word {
                JoltInstructionKind::VirtualChangeDivisorW
            } else {
                JoltInstructionKind::VirtualChangeDivisor
            },
            t0,
            dividend,
            divisor,
        )),
    ]);
    state.materialize_ops_into(&mut sequence, instruction, ops)?;

    if word {
        state.materialize_ops_into(
            &mut sequence,
            instruction,
            [
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::VirtualSignExtendWord,
                    t1,
                    a2,
                    0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertEQ,
                    t1,
                    a2,
                    0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::SRAI,
                    t2,
                    a3,
                    32,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertEQ,
                    t2,
                    0,
                    0,
                )),
            ],
        )?;
    } else {
        state.materialize_ops_into(
            &mut sequence,
            instruction,
            [grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::MULH,
                t1,
                a2,
                t0,
            ))],
        )?;
        t2 = state.allocator().allocate()?;
        t3 = state.allocator().allocate()?;
        state.materialize_ops_into(
            &mut sequence,
            instruction,
            [
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::MUL,
                    t2,
                    a2,
                    t0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::SRAI,
                    t3,
                    t2,
                    shmat,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertEQ,
                    t1,
                    t3,
                    0,
                )),
            ],
        )?;
    }

    if word {
        state.materialize_ops_into(
            &mut sequence,
            instruction,
            [
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::SRAI,
                    t2,
                    dividend,
                    shmat,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::XOR,
                    t3,
                    a3,
                    t2,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::SUB,
                    t3,
                    t3,
                    t2,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::MUL,
                    t1,
                    a2,
                    t0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::ADD,
                    t1,
                    t1,
                    t3,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertEQ,
                    t1,
                    dividend,
                    0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::SRAI,
                    t2,
                    t0,
                    31,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::XOR,
                    t1,
                    t0,
                    t2,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::SUB,
                    t1,
                    t1,
                    t2,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
                    a3,
                    t1,
                    0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::VirtualSignExtendWord,
                    rd(instruction)?,
                    if remainder_output { t3 } else { a2 },
                    0,
                )),
            ],
        )?;
    } else {
        let abs_divisor = if remainder_output { t2 } else { t3 };
        state.materialize_ops_into(
            &mut sequence,
            instruction,
            [
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::SRAI,
                    t1,
                    dividend,
                    shmat,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::XOR,
                    t3,
                    a3,
                    t1,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::SUB,
                    t3,
                    t3,
                    t1,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::ADD,
                    t2,
                    t2,
                    t3,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertEQ,
                    t2,
                    a0,
                    0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::SRAI,
                    t1,
                    t0,
                    shmat,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::XOR,
                    abs_divisor,
                    t0,
                    t1,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                    JoltInstructionKind::SUB,
                    abs_divisor,
                    abs_divisor,
                    t1,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                    JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
                    a3,
                    abs_divisor,
                    0,
                )),
                grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                    JoltInstructionKind::ADDI,
                    rd(instruction)?,
                    if remainder_output { t3 } else { a2 },
                    0,
                )),
            ],
        )?;
    }

    let mut released = vec![a2, a3, t0, t1, t2, t3];
    if let Some(t4) = t4 {
        released.push(t4);
    }
    state.materialize_ops_into(
        &mut sequence,
        instruction,
        released.into_iter().map(grammar::ExpansionOp::Release),
    )?;
    sequence.finish()
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

    let mut ops = vec![
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualZeroExtendWord,
            rs1_extended,
            rs1(instruction)?,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualZeroExtendWord,
            rs2_extended,
            rs2(instruction)?,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::j(
            JoltInstructionKind::VirtualAdvice,
            quotient,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
            JoltInstructionKind::VirtualAssertMulUNoOverflow,
            quotient,
            rs2_extended,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::MUL,
            tmp,
            quotient,
            rs2_extended,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
            JoltInstructionKind::VirtualAssertLTE,
            tmp,
            rs1_extended,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SUB,
            tmp,
            rs1_extended,
            tmp,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
            JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
            tmp,
            rs2_extended,
            0,
        )),
    ];

    if remainder_output {
        ops.push(grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            tmp,
            0,
        )));
    } else {
        ops.extend([
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::VirtualSignExtendWord,
                tmp,
                quotient,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::b(
                JoltInstructionKind::VirtualAssertValidDiv0,
                rs2_extended,
                tmp,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                tmp,
                0,
            )),
        ]);
    }

    ops.extend([
        grammar::ExpansionOp::Release(rs1_extended),
        grammar::ExpansionOp::Release(rs2_extended),
        grammar::ExpansionOp::Release(quotient),
    ]);
    if !remainder_output {
        ops.push(grammar::ExpansionOp::Release(tmp));
    }
    core::ExpansionState::new(allocator).materialize_ops(instruction, ops)
}
