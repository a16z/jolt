use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_ram_region_assertion(
    asm: &mut ExpansionBuilder,
    address_register: RegisterOperand,
    ram_start: TempId,
) -> Result<(), ExpansionError> {
    asm.expand_u(
        JoltInstructionKind::LUI,
        ram_start.operand(),
        RAM_START_ADDRESS as i128,
    );
    asm.expand_b(
        JoltInstructionKind::VirtualAssertLTE,
        ram_start.operand(),
        address_register,
        0,
    );
    asm.release(ram_start);
    Ok(())
}

pub(in crate::expand) fn expand_byte_load(
    instruction: &JoltRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        JoltInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(JoltInstructionKind::LD, v1.operand(), v1.operand(), 0);
    asm.expand_i(JoltInstructionKind::XORI, v0.operand(), v0.operand(), 7);
    asm.expand_i(JoltInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        JoltInstructionKind::SLL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        if signed {
            JoltInstructionKind::SRAI
        } else {
            JoltInstructionKind::SRLI
        },
        reg(rd(instruction)?),
        v1.operand(),
        56,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}

pub(in crate::expand) fn expand_halfword_load(
    instruction: &JoltRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_address(
        JoltInstructionKind::VirtualAssertHalfwordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        JoltInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(JoltInstructionKind::LD, v1.operand(), v1.operand(), 0);
    asm.expand_i(JoltInstructionKind::XORI, v0.operand(), v0.operand(), 6);
    asm.expand_i(JoltInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        JoltInstructionKind::SLL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        if signed {
            JoltInstructionKind::SRAI
        } else {
            JoltInstructionKind::SRLI
        },
        reg(rd(instruction)?),
        v1.operand(),
        48,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}

pub(in crate::expand) fn expand_advice_load(
    instruction: &JoltRow,
    byte_len: i128,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.expand_j(
        JoltInstructionKind::VirtualAdviceLoad,
        reg(rd(instruction)?),
        byte_len,
    );
    if byte_len < 8 {
        let shift = 64 - byte_len * 8;
        asm.expand_i(
            JoltInstructionKind::SLLI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
        asm.expand_i(
            JoltInstructionKind::SRAI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
    }

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_d(
    instruction: &JoltRow,
    op: JoltInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs2 = asm.allocate()?;
    let v_rd = asm.allocate()?;

    asm.expand_i(
        JoltInstructionKind::LD,
        v_rd.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(op, v_rs2.operand(), v_rd.operand(), reg(rs2(instruction)?));
    asm.expand_s(
        JoltInstructionKind::SD,
        reg(rs1(instruction)?),
        v_rs2.operand(),
        0,
    );
    asm.expand_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        v_rd.operand(),
        0,
    );
    asm.release_many([v_rs2, v_rd]);

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_minmax_d(
    instruction: &JoltRow,
    compare_op: JoltInstructionKind,
    min: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let (cmp_rs1, cmp_rs2): (RegisterOperand, RegisterOperand) = if min {
        (reg(rs2(instruction)?), v0.operand())
    } else {
        (v0.operand(), reg(rs2(instruction)?))
    };

    asm.expand_i(
        JoltInstructionKind::LD,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(compare_op, v1.operand(), cmp_rs1, cmp_rs2);
    asm.expand_r(
        JoltInstructionKind::SUB,
        v2.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::MUL,
        v2.operand(),
        v2.operand(),
        v1.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::ADD,
        v1.operand(),
        v0.operand(),
        v2.operand(),
    );
    asm.expand_s(
        JoltInstructionKind::SD,
        reg(rs1(instruction)?),
        v1.operand(),
        0,
    );
    asm.expand_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release_many([v0, v1, v2]);

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_w(
    instruction: &JoltRow,
    op: JoltInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;
    let v_rs2 = asm.allocate()?;
    let v_mask = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;

    expand_amo_pre64(
        &mut asm,
        reg(rs1(instruction)?),
        v_rd.operand(),
        v_dword.operand(),
        v_shift.operand(),
    )?;
    asm.expand_r(op, v_rs2.operand(), v_rd.operand(), reg(rs2(instruction)?));
    expand_amo_post64(
        &mut asm,
        AmoPost64 {
            rs1: reg(rs1(instruction)?),
            v_rs2: v_rs2.operand(),
            v_dword: v_dword.operand(),
            v_shift: v_shift.operand(),
            v_mask: v_mask.operand(),
            rd: reg(rd(instruction)?),
            v_rd: v_rd.operand(),
        },
    )?;
    asm.release_many([v_rd, v_rs2, v_mask, v_dword, v_shift]);

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_minmax_w(
    instruction: &JoltRow,
    compare_op: JoltInstructionKind,
    min: bool,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;

    expand_amo_pre64(
        &mut asm,
        reg(rs1(instruction)?),
        v_rd.operand(),
        v_dword.operand(),
        v_shift.operand(),
    )?;

    let v_rs2 = asm.allocate()?;
    let v0 = asm.allocate()?;
    let extend_op = if signed {
        JoltInstructionKind::VirtualSignExtendWord
    } else {
        JoltInstructionKind::VirtualZeroExtendWord
    };
    asm.expand_i(extend_op, v_rs2.operand(), reg(rs2(instruction)?), 0);
    asm.expand_i(extend_op, v0.operand(), v_rd.operand(), 0);
    let (cmp_rs1, cmp_rs2) = if min {
        (v_rs2.operand(), v0.operand())
    } else {
        (v0.operand(), v_rs2.operand())
    };
    asm.expand_r(compare_op, v0.operand(), cmp_rs1, cmp_rs2);
    asm.expand_r(
        JoltInstructionKind::SUB,
        v_rs2.operand(),
        reg(rs2(instruction)?),
        v_rd.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::MUL,
        v_rs2.operand(),
        v_rs2.operand(),
        v0.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::ADD,
        v_rs2.operand(),
        v_rs2.operand(),
        v_rd.operand(),
    );
    expand_amo_post64(
        &mut asm,
        AmoPost64 {
            rs1: reg(rs1(instruction)?),
            v_rs2: v_rs2.operand(),
            v_dword: v_dword.operand(),
            v_shift: v_shift.operand(),
            v_mask: v0.operand(),
            rd: reg(rd(instruction)?),
            v_rd: v_rd.operand(),
        },
    )?;
    asm.release_many([v_rd, v_dword, v_shift, v_rs2, v0]);

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_pre64(
    asm: &mut ExpansionBuilder,
    rs1: RegisterOperand,
    v_rd: RegisterOperand,
    v_dword: RegisterOperand,
    v_shift: RegisterOperand,
) -> Result<(), ExpansionError> {
    asm.expand_address(JoltInstructionKind::VirtualAssertWordAlignment, rs1, 0);
    asm.expand_i(JoltInstructionKind::ANDI, v_shift, rs1, format_i_imm(-8));
    asm.expand_i(JoltInstructionKind::LD, v_dword, v_shift, 0);
    asm.expand_i(JoltInstructionKind::SLLI, v_shift, rs1, 3);
    asm.expand_r(JoltInstructionKind::SRL, v_rd, v_dword, v_shift);
    Ok(())
}

pub(in crate::expand) struct AmoPost64 {
    pub(in crate::expand) rs1: RegisterOperand,
    pub(in crate::expand) v_rs2: RegisterOperand,
    pub(in crate::expand) v_dword: RegisterOperand,
    pub(in crate::expand) v_shift: RegisterOperand,
    pub(in crate::expand) v_mask: RegisterOperand,
    pub(in crate::expand) rd: RegisterOperand,
    pub(in crate::expand) v_rd: RegisterOperand,
}

pub(in crate::expand) fn expand_amo_post64(
    asm: &mut ExpansionBuilder,
    registers: AmoPost64,
) -> Result<(), ExpansionError> {
    let AmoPost64 {
        rs1,
        v_rs2,
        v_dword,
        v_shift,
        v_mask,
        rd,
        v_rd,
    } = registers;

    asm.expand_i(JoltInstructionKind::ORI, v_mask, reg(0), format_i_imm(-1));
    asm.expand_i(JoltInstructionKind::SRLI, v_mask, v_mask, 32);
    asm.expand_r(JoltInstructionKind::SLL, v_mask, v_mask, v_shift);
    asm.expand_r(JoltInstructionKind::SLL, v_shift, v_rs2, v_shift);
    asm.expand_r(JoltInstructionKind::XOR, v_shift, v_dword, v_shift);
    asm.expand_r(JoltInstructionKind::AND, v_shift, v_shift, v_mask);
    asm.expand_r(JoltInstructionKind::XOR, v_dword, v_dword, v_shift);
    asm.expand_i(JoltInstructionKind::ANDI, v_mask, rs1, format_i_imm(-8));
    asm.expand_s(JoltInstructionKind::SD, v_mask, v_dword, 0);
    asm.expand_i(JoltInstructionKind::VirtualSignExtendWord, rd, v_rd, 0);
    Ok(())
}

pub(in crate::expand) fn expand_narrow_store(
    instruction: &JoltRow,
    mask: i128,
    alignment: Option<JoltInstructionKind>,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    if let Some(alignment) = alignment {
        asm.expand_address(alignment, reg(rs1(instruction)?), instruction.operands.imm);
    }
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        JoltInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(JoltInstructionKind::LD, v2.operand(), v1.operand(), 0);
    asm.expand_i(JoltInstructionKind::SLLI, v3.operand(), v0.operand(), 3);
    asm.expand_u(JoltInstructionKind::LUI, v0.operand(), mask);
    asm.expand_r(
        JoltInstructionKind::SLL,
        v0.operand(),
        v0.operand(),
        v3.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::SLL,
        v3.operand(),
        reg(rs2(instruction)?),
        v3.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::XOR,
        v3.operand(),
        v2.operand(),
        v3.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::AND,
        v3.operand(),
        v3.operand(),
        v0.operand(),
    );
    asm.expand_r(
        JoltInstructionKind::XOR,
        v2.operand(),
        v2.operand(),
        v3.operand(),
    );
    asm.expand_s(JoltInstructionKind::SD, v1.operand(), v2.operand(), 0);
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
