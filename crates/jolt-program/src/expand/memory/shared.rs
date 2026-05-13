use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_ram_region_assertion(
    asm: &mut ExpansionBuilder,
    address_register: RegisterOperand,
    ram_start: TempId,
) -> Result<(), ExpansionError> {
    asm.expand_u(
        SourceInstructionKind::LUI,
        ram_start.operand(),
        RAM_START_ADDRESS as i128,
    );
    asm.expand_b(
        SourceInstructionKind::VirtualAssertLTE,
        ram_start.operand(),
        address_register,
        0,
    );
    asm.release(ram_start);
    Ok(())
}

pub(in crate::expand) fn expand_byte_load(
    instruction: &SourceRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_i(
        SourceInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        SourceInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(SourceInstructionKind::LD, v1.operand(), v1.operand(), 0);
    asm.expand_i(SourceInstructionKind::XORI, v0.operand(), v0.operand(), 7);
    asm.expand_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        SourceInstructionKind::SLL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        if signed {
            SourceInstructionKind::SRAI
        } else {
            SourceInstructionKind::SRLI
        },
        reg(rd(instruction)?),
        v1.operand(),
        56,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}

pub(in crate::expand) fn expand_halfword_load(
    instruction: &SourceRow,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_address(
        SourceInstructionKind::VirtualAssertHalfwordAlignment,
        reg(rs1(instruction)?),
        instruction.operands.imm,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        SourceInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(SourceInstructionKind::LD, v1.operand(), v1.operand(), 0);
    asm.expand_i(SourceInstructionKind::XORI, v0.operand(), v0.operand(), 6);
    asm.expand_i(SourceInstructionKind::SLLI, v0.operand(), v0.operand(), 3);
    asm.expand_r(
        SourceInstructionKind::SLL,
        v1.operand(),
        v1.operand(),
        v0.operand(),
    );
    asm.expand_i(
        if signed {
            SourceInstructionKind::SRAI
        } else {
            SourceInstructionKind::SRLI
        },
        reg(rd(instruction)?),
        v1.operand(),
        48,
    );
    asm.release_many([v0, v1]);

    asm.finalize()
}

pub(in crate::expand) fn expand_advice_load(
    instruction: &SourceRow,
    byte_len: i128,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.expand_j(
        SourceInstructionKind::VirtualAdviceLoad,
        reg(rd(instruction)?),
        byte_len,
    );
    if byte_len < 8 {
        let shift = 64 - byte_len * 8;
        asm.expand_i(
            SourceInstructionKind::SLLI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
        asm.expand_i(
            SourceInstructionKind::SRAI,
            reg(rd(instruction)?),
            reg(rd(instruction)?),
            shift,
        );
    }

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_d(
    instruction: &SourceRow,
    op: SourceInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs2 = asm.allocate()?;
    let v_rd = asm.allocate()?;

    asm.expand_i(
        SourceInstructionKind::LD,
        v_rd.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(op, v_rs2.operand(), v_rd.operand(), reg(rs2(instruction)?));
    asm.expand_s(
        SourceInstructionKind::SD,
        reg(rs1(instruction)?),
        v_rs2.operand(),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v_rd.operand(),
        0,
    );
    asm.release_many([v_rs2, v_rd]);

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_minmax_d(
    instruction: &SourceRow,
    compare_op: SourceInstructionKind,
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
        SourceInstructionKind::LD,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_r(compare_op, v1.operand(), cmp_rs1, cmp_rs2);
    asm.expand_r(
        SourceInstructionKind::SUB,
        v2.operand(),
        reg(rs2(instruction)?),
        v0.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v2.operand(),
        v2.operand(),
        v1.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::ADD,
        v1.operand(),
        v0.operand(),
        v2.operand(),
    );
    asm.expand_s(
        SourceInstructionKind::SD,
        reg(rs1(instruction)?),
        v1.operand(),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v0.operand(),
        0,
    );
    asm.release_many([v0, v1, v2]);

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_w(
    instruction: &SourceRow,
    op: SourceInstructionKind,
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
    instruction: &SourceRow,
    compare_op: SourceInstructionKind,
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
        SourceInstructionKind::VirtualSignExtendWord
    } else {
        SourceInstructionKind::VirtualZeroExtendWord
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
        SourceInstructionKind::SUB,
        v_rs2.operand(),
        reg(rs2(instruction)?),
        v_rd.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::MUL,
        v_rs2.operand(),
        v_rs2.operand(),
        v0.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::ADD,
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
    asm.expand_address(SourceInstructionKind::VirtualAssertWordAlignment, rs1, 0);
    asm.expand_i(SourceInstructionKind::ANDI, v_shift, rs1, format_i_imm(-8));
    asm.expand_i(SourceInstructionKind::LD, v_dword, v_shift, 0);
    asm.expand_i(SourceInstructionKind::SLLI, v_shift, rs1, 3);
    asm.expand_r(SourceInstructionKind::SRL, v_rd, v_dword, v_shift);
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

    asm.expand_i(SourceInstructionKind::ORI, v_mask, reg(0), format_i_imm(-1));
    asm.expand_i(SourceInstructionKind::SRLI, v_mask, v_mask, 32);
    asm.expand_r(SourceInstructionKind::SLL, v_mask, v_mask, v_shift);
    asm.expand_r(SourceInstructionKind::SLL, v_shift, v_rs2, v_shift);
    asm.expand_r(SourceInstructionKind::XOR, v_shift, v_dword, v_shift);
    asm.expand_r(SourceInstructionKind::AND, v_shift, v_shift, v_mask);
    asm.expand_r(SourceInstructionKind::XOR, v_dword, v_dword, v_shift);
    asm.expand_i(SourceInstructionKind::ANDI, v_mask, rs1, format_i_imm(-8));
    asm.expand_s(SourceInstructionKind::SD, v_mask, v_dword, 0);
    asm.expand_i(SourceInstructionKind::VirtualSignExtendWord, rd, v_rd, 0);
    Ok(())
}

pub(in crate::expand) fn expand_narrow_store(
    instruction: &SourceRow,
    mask: i128,
    alignment: Option<SourceInstructionKind>,
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
        SourceInstructionKind::ADDI,
        v0.operand(),
        reg(rs1(instruction)?),
        format_i_imm(instruction.operands.imm),
    );
    asm.expand_i(
        SourceInstructionKind::ANDI,
        v1.operand(),
        v0.operand(),
        format_i_imm(-8),
    );
    asm.expand_i(SourceInstructionKind::LD, v2.operand(), v1.operand(), 0);
    asm.expand_i(SourceInstructionKind::SLLI, v3.operand(), v0.operand(), 3);
    asm.expand_u(SourceInstructionKind::LUI, v0.operand(), mask);
    asm.expand_r(
        SourceInstructionKind::SLL,
        v0.operand(),
        v0.operand(),
        v3.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::SLL,
        v3.operand(),
        reg(rs2(instruction)?),
        v3.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::XOR,
        v3.operand(),
        v2.operand(),
        v3.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::AND,
        v3.operand(),
        v3.operand(),
        v0.operand(),
    );
    asm.expand_r(
        SourceInstructionKind::XOR,
        v2.operand(),
        v2.operand(),
        v3.operand(),
    );
    asm.expand_s(SourceInstructionKind::SD, v1.operand(), v2.operand(), 0);
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
