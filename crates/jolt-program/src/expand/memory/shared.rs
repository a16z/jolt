use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn emit_ram_region_assertion(
    sequence: &mut core::ExpansionSequence,
    address_register: u8,
    allocator: &mut ExpansionAllocator,
) -> Result<(), ExpansionError> {
    let ram_start = allocator.allocate()?;
    sequence.emit_u_expanded(
        JoltInstructionKind::LUI,
        ram_start,
        RAM_START_ADDRESS as i128,
        allocator,
    )?;
    sequence.emit_b_expanded(
        JoltInstructionKind::VirtualAssertLTE,
        ram_start,
        address_register,
        0,
        allocator,
    )?;
    allocator.release(ram_start)?;
    Ok(())
}

pub(in crate::expand) fn expand_byte_load(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    signed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v0,
                rs1(instruction)?,
                format_i_imm(instruction.operands.imm),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ANDI,
                v1,
                v0,
                format_i_imm(-8),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                v1,
                v1,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::XORI,
                v0,
                v0,
                7,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SLLI,
                v0,
                v0,
                3,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SLL,
                v1,
                v1,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                if signed {
                    JoltInstructionKind::SRAI
                } else {
                    JoltInstructionKind::SRLI
                },
                rd(instruction)?,
                v1,
                56,
            )),
            grammar::ExpansionOp::Release(v0),
            grammar::ExpansionOp::Release(v1),
        ],
    )
}

pub(in crate::expand) fn expand_halfword_load(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    signed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::address(
                JoltInstructionKind::VirtualAssertHalfwordAlignment,
                rs1(instruction)?,
                instruction.operands.imm,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                v0,
                rs1(instruction)?,
                format_i_imm(instruction.operands.imm),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ANDI,
                v1,
                v0,
                format_i_imm(-8),
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                v1,
                v1,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::XORI,
                v0,
                v0,
                6,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SLLI,
                v0,
                v0,
                3,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SLL,
                v1,
                v1,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                if signed {
                    JoltInstructionKind::SRAI
                } else {
                    JoltInstructionKind::SRLI
                },
                rd(instruction)?,
                v1,
                48,
            )),
            grammar::ExpansionOp::Release(v0),
            grammar::ExpansionOp::Release(v1),
        ],
    )
}

pub(in crate::expand) fn expand_advice_load(
    instruction: &NormalizedInstruction,
    byte_len: i128,
    sign_extension_shift: Option<i128>,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut ops = vec![grammar::ExpansionOp::Expand(grammar::RowTemplate::j(
        JoltInstructionKind::VirtualAdviceLoad,
        rd(instruction)?,
        byte_len,
    ))];
    if let Some(shift) = sign_extension_shift {
        ops.extend([
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SLLI,
                rd(instruction)?,
                rd(instruction)?,
                shift,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::SRAI,
                rd(instruction)?,
                rd(instruction)?,
                shift,
            )),
        ]);
    }
    core::ExpansionState::new(allocator).materialize_ops(instruction, ops)
}

pub(in crate::expand) fn expand_amo_d(
    instruction: &NormalizedInstruction,
    op: JoltInstructionKind,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs2 = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                v_rd,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                op,
                v_rs2,
                v_rd,
                rs2(instruction)?,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::s(
                JoltInstructionKind::SD,
                rs1(instruction)?,
                v_rs2,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                v_rd,
                0,
            )),
            grammar::ExpansionOp::Release(v_rs2),
            grammar::ExpansionOp::Release(v_rd),
        ],
    )
}

pub(in crate::expand) fn expand_amo_minmax_d(
    instruction: &NormalizedInstruction,
    compare_op: JoltInstructionKind,
    min: bool,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let (cmp_rs1, cmp_rs2) = if min {
        (rs2(instruction)?, v0)
    } else {
        (v0, rs2(instruction)?)
    };
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::LD,
                v0,
                rs1(instruction)?,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(compare_op, v1, cmp_rs1, cmp_rs2)),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::SUB,
                v2,
                rs2(instruction)?,
                v0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::MUL,
                v2,
                v2,
                v1,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
                JoltInstructionKind::ADD,
                v1,
                v0,
                v2,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::s(
                JoltInstructionKind::SD,
                rs1(instruction)?,
                v1,
                0,
            )),
            grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                v0,
                0,
            )),
            grammar::ExpansionOp::Release(v0),
            grammar::ExpansionOp::Release(v1),
            grammar::ExpansionOp::Release(v2),
        ],
    )
}

pub(in crate::expand) fn expand_amo_w(
    instruction: &NormalizedInstruction,
    op: JoltInstructionKind,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let v_rs2 = allocator.allocate()?;
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let mut ops = amo_pre64_ops(rs1(instruction)?, v_rd, v_dword, v_shift);
    ops.push(grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
        op,
        v_rs2,
        v_rd,
        rs2(instruction)?,
    )));
    ops.extend(amo_post64_ops(
        rs1(instruction)?,
        v_rs2,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
    ));
    ops.extend([
        grammar::ExpansionOp::Release(v_rd),
        grammar::ExpansionOp::Release(v_rs2),
        grammar::ExpansionOp::Release(v_mask),
        grammar::ExpansionOp::Release(v_dword),
        grammar::ExpansionOp::Release(v_shift),
    ]);
    core::ExpansionState::new(allocator).materialize_ops(instruction, ops)
}

pub(in crate::expand) fn expand_amo_minmax_w(
    instruction: &NormalizedInstruction,
    compare_op: JoltInstructionKind,
    min: bool,
    signed: bool,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let mut ops = amo_pre64_ops(rs1(instruction)?, v_rd, v_dword, v_shift);

    let v_rs2 = allocator.allocate()?;
    let v0 = allocator.allocate()?;
    let extend_op = if signed {
        JoltInstructionKind::VirtualSignExtendWord
    } else {
        JoltInstructionKind::VirtualZeroExtendWord
    };
    ops.extend([
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            extend_op,
            v_rs2,
            rs2(instruction)?,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(extend_op, v0, v_rd, 0)),
    ]);
    let (cmp_rs1, cmp_rs2) = if min { (v_rs2, v0) } else { (v0, v_rs2) };
    ops.extend([
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(compare_op, v0, cmp_rs1, cmp_rs2)),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SUB,
            v_rs2,
            rs2(instruction)?,
            v_rd,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::MUL,
            v_rs2,
            v_rs2,
            v0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::ADD,
            v_rs2,
            v_rs2,
            v_rd,
        )),
    ]);
    ops.extend(amo_post64_ops(
        rs1(instruction)?,
        v_rs2,
        v_dword,
        v_shift,
        v0,
        rd(instruction)?,
        v_rd,
    ));
    ops.extend([
        grammar::ExpansionOp::Release(v_rd),
        grammar::ExpansionOp::Release(v_dword),
        grammar::ExpansionOp::Release(v_shift),
        grammar::ExpansionOp::Release(v_rs2),
        grammar::ExpansionOp::Release(v0),
    ]);
    core::ExpansionState::new(allocator).materialize_ops(instruction, ops)
}

pub(in crate::expand) fn amo_pre64_ops(
    rs1: u8,
    v_rd: u8,
    v_dword: u8,
    v_shift: u8,
) -> Vec<grammar::ExpansionOp> {
    vec![
        grammar::ExpansionOp::Expand(grammar::RowTemplate::address(
            JoltInstructionKind::VirtualAssertWordAlignment,
            rs1,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::ANDI,
            v_shift,
            rs1,
            format_i_imm(-8),
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::LD,
            v_dword,
            v_shift,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::SLLI,
            v_shift,
            rs1,
            3,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SRL,
            v_rd,
            v_dword,
            v_shift,
        )),
    ]
}

pub(in crate::expand) fn amo_post64_ops(
    rs1: u8,
    v_rs2: u8,
    v_dword: u8,
    v_shift: u8,
    v_mask: u8,
    rd: u8,
    v_rd: u8,
) -> Vec<grammar::ExpansionOp> {
    vec![
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::ORI,
            v_mask,
            0,
            format_i_imm(-1),
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::SRLI,
            v_mask,
            v_mask,
            32,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SLL,
            v_mask,
            v_mask,
            v_shift,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SLL,
            v_shift,
            v_rs2,
            v_shift,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::XOR,
            v_shift,
            v_dword,
            v_shift,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::AND,
            v_shift,
            v_shift,
            v_mask,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::XOR,
            v_dword,
            v_dword,
            v_shift,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::ANDI,
            v_mask,
            rs1,
            format_i_imm(-8),
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::s(
            JoltInstructionKind::SD,
            v_mask,
            v_dword,
            0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualSignExtendWord,
            rd,
            v_rd,
            0,
        )),
    ]
}

pub(in crate::expand) fn expand_narrow_store(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    mask: i128,
    alignment: Option<JoltInstructionKind>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut ops = Vec::new();
    if let Some(alignment) = alignment {
        ops.push(grammar::ExpansionOp::Expand(grammar::RowTemplate::address(
            alignment,
            rs1(instruction)?,
            instruction.operands.imm,
        )));
    }
    ops.extend([
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::ADDI,
            v0,
            rs1(instruction)?,
            format_i_imm(instruction.operands.imm),
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::ANDI,
            v1,
            v0,
            format_i_imm(-8),
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(JoltInstructionKind::LD, v2, v1, 0)),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::i(
            JoltInstructionKind::SLLI,
            v3,
            v0,
            3,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::u(JoltInstructionKind::LUI, v0, mask)),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SLL,
            v0,
            v0,
            v3,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::SLL,
            v3,
            rs2(instruction)?,
            v3,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::XOR,
            v3,
            v2,
            v3,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::AND,
            v3,
            v3,
            v0,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::r(
            JoltInstructionKind::XOR,
            v2,
            v2,
            v3,
        )),
        grammar::ExpansionOp::Expand(grammar::RowTemplate::s(JoltInstructionKind::SD, v1, v2, 0)),
        grammar::ExpansionOp::Release(v0),
        grammar::ExpansionOp::Release(v1),
        grammar::ExpansionOp::Release(v2),
        grammar::ExpansionOp::Release(v3),
    ]);
    core::ExpansionState::new(allocator).materialize_ops(instruction, ops)
}
