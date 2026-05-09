use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn expand_ram_region_assertion(
    asm: &mut ExpansionBuilder,
    address_register: u8,
    ram_start: u8,
) -> Result<(), ExpansionError> {
    asm.expand_u(
        JoltInstructionKind::LUI,
        ram_start,
        RAM_START_ADDRESS as i128,
    )?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertLTE,
        ram_start,
        address_register,
        0,
    )?;
    asm.release(ram_start)
}

pub(in crate::expand) fn expand_byte_load(
    instruction: &NormalizedInstruction,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.expand_i(JoltInstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.expand_i(JoltInstructionKind::LD, v1, v1, 0)?;
    asm.expand_i(JoltInstructionKind::XORI, v0, v0, 7)?;
    asm.expand_i(JoltInstructionKind::SLLI, v0, v0, 3)?;
    asm.expand_r(JoltInstructionKind::SLL, v1, v1, v0)?;
    asm.expand_i(
        if signed {
            JoltInstructionKind::SRAI
        } else {
            JoltInstructionKind::SRLI
        },
        rd(instruction)?,
        v1,
        56,
    )?;
    asm.release_many([v0, v1])?;

    asm.finalize()
}

pub(in crate::expand) fn expand_halfword_load(
    instruction: &NormalizedInstruction,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_address(
        JoltInstructionKind::VirtualAssertHalfwordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.expand_i(JoltInstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.expand_i(JoltInstructionKind::LD, v1, v1, 0)?;
    asm.expand_i(JoltInstructionKind::XORI, v0, v0, 6)?;
    asm.expand_i(JoltInstructionKind::SLLI, v0, v0, 3)?;
    asm.expand_r(JoltInstructionKind::SLL, v1, v1, v0)?;
    asm.expand_i(
        if signed {
            JoltInstructionKind::SRAI
        } else {
            JoltInstructionKind::SRLI
        },
        rd(instruction)?,
        v1,
        48,
    )?;
    asm.release_many([v0, v1])?;

    asm.finalize()
}

pub(in crate::expand) fn expand_advice_load(
    instruction: &NormalizedInstruction,
    byte_len: i128,
    sign_extension_shift: Option<i128>,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.expand_j(
        JoltInstructionKind::VirtualAdviceLoad,
        rd(instruction)?,
        byte_len,
    )?;
    if let Some(shift) = sign_extension_shift {
        asm.expand_i(
            JoltInstructionKind::SLLI,
            rd(instruction)?,
            rd(instruction)?,
            shift,
        )?;
        asm.expand_i(
            JoltInstructionKind::SRAI,
            rd(instruction)?,
            rd(instruction)?,
            shift,
        )?;
    }

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_d(
    instruction: &NormalizedInstruction,
    op: JoltInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs2 = asm.allocate()?;
    let v_rd = asm.allocate()?;

    asm.expand_i(JoltInstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.expand_r(op, v_rs2, v_rd, rs2(instruction)?)?;
    asm.expand_s(JoltInstructionKind::SD, rs1(instruction)?, v_rs2, 0)?;
    asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    asm.release_many([v_rs2, v_rd])?;

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_minmax_d(
    instruction: &NormalizedInstruction,
    compare_op: JoltInstructionKind,
    min: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let (cmp_rs1, cmp_rs2) = if min {
        (rs2(instruction)?, v0)
    } else {
        (v0, rs2(instruction)?)
    };

    asm.expand_i(JoltInstructionKind::LD, v0, rs1(instruction)?, 0)?;
    asm.expand_r(compare_op, v1, cmp_rs1, cmp_rs2)?;
    asm.expand_r(JoltInstructionKind::SUB, v2, rs2(instruction)?, v0)?;
    asm.expand_r(JoltInstructionKind::MUL, v2, v2, v1)?;
    asm.expand_r(JoltInstructionKind::ADD, v1, v0, v2)?;
    asm.expand_s(JoltInstructionKind::SD, rs1(instruction)?, v1, 0)?;
    asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    asm.release_many([v0, v1, v2])?;

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_w(
    instruction: &NormalizedInstruction,
    op: JoltInstructionKind,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;
    let v_rs2 = asm.allocate()?;
    let v_mask = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;

    expand_amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    asm.expand_r(op, v_rs2, v_rd, rs2(instruction)?)?;
    expand_amo_post64(
        &mut asm,
        AmoPost64 {
            rs1: rs1(instruction)?,
            v_rs2,
            v_dword,
            v_shift,
            v_mask,
            rd: rd(instruction)?,
            v_rd,
        },
    )?;
    asm.release_many([v_rd, v_rs2, v_mask, v_dword, v_shift])?;

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_minmax_w(
    instruction: &NormalizedInstruction,
    compare_op: JoltInstructionKind,
    min: bool,
    signed: bool,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;
    let v_dword = asm.allocate()?;
    let v_shift = asm.allocate()?;

    expand_amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;

    let v_rs2 = asm.allocate()?;
    let v0 = asm.allocate()?;
    let extend_op = if signed {
        JoltInstructionKind::VirtualSignExtendWord
    } else {
        JoltInstructionKind::VirtualZeroExtendWord
    };
    asm.expand_i(extend_op, v_rs2, rs2(instruction)?, 0)?;
    asm.expand_i(extend_op, v0, v_rd, 0)?;
    let (cmp_rs1, cmp_rs2) = if min { (v_rs2, v0) } else { (v0, v_rs2) };
    asm.expand_r(compare_op, v0, cmp_rs1, cmp_rs2)?;
    asm.expand_r(JoltInstructionKind::SUB, v_rs2, rs2(instruction)?, v_rd)?;
    asm.expand_r(JoltInstructionKind::MUL, v_rs2, v_rs2, v0)?;
    asm.expand_r(JoltInstructionKind::ADD, v_rs2, v_rs2, v_rd)?;
    expand_amo_post64(
        &mut asm,
        AmoPost64 {
            rs1: rs1(instruction)?,
            v_rs2,
            v_dword,
            v_shift,
            v_mask: v0,
            rd: rd(instruction)?,
            v_rd,
        },
    )?;
    asm.release_many([v_rd, v_dword, v_shift, v_rs2, v0])?;

    asm.finalize()
}

pub(in crate::expand) fn expand_amo_pre64(
    asm: &mut ExpansionBuilder,
    rs1: u8,
    v_rd: u8,
    v_dword: u8,
    v_shift: u8,
) -> Result<(), ExpansionError> {
    asm.expand_address(JoltInstructionKind::VirtualAssertWordAlignment, rs1, 0)?;
    asm.expand_i(JoltInstructionKind::ANDI, v_shift, rs1, format_i_imm(-8))?;
    asm.expand_i(JoltInstructionKind::LD, v_dword, v_shift, 0)?;
    asm.expand_i(JoltInstructionKind::SLLI, v_shift, rs1, 3)?;
    asm.expand_r(JoltInstructionKind::SRL, v_rd, v_dword, v_shift)
}

pub(in crate::expand) struct AmoPost64 {
    pub(in crate::expand) rs1: u8,
    pub(in crate::expand) v_rs2: u8,
    pub(in crate::expand) v_dword: u8,
    pub(in crate::expand) v_shift: u8,
    pub(in crate::expand) v_mask: u8,
    pub(in crate::expand) rd: u8,
    pub(in crate::expand) v_rd: u8,
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

    asm.expand_i(JoltInstructionKind::ORI, v_mask, 0, format_i_imm(-1))?;
    asm.expand_i(JoltInstructionKind::SRLI, v_mask, v_mask, 32)?;
    asm.expand_r(JoltInstructionKind::SLL, v_mask, v_mask, v_shift)?;
    asm.expand_r(JoltInstructionKind::SLL, v_shift, v_rs2, v_shift)?;
    asm.expand_r(JoltInstructionKind::XOR, v_shift, v_dword, v_shift)?;
    asm.expand_r(JoltInstructionKind::AND, v_shift, v_shift, v_mask)?;
    asm.expand_r(JoltInstructionKind::XOR, v_dword, v_dword, v_shift)?;
    asm.expand_i(JoltInstructionKind::ANDI, v_mask, rs1, format_i_imm(-8))?;
    asm.expand_s(JoltInstructionKind::SD, v_mask, v_dword, 0)?;
    asm.expand_i(JoltInstructionKind::VirtualSignExtendWord, rd, v_rd, 0)
}

pub(in crate::expand) fn expand_narrow_store(
    instruction: &NormalizedInstruction,
    mask: i128,
    alignment: Option<JoltInstructionKind>,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    if let Some(alignment) = alignment {
        asm.expand_address(alignment, rs1(instruction)?, instruction.operands.imm)?;
    }
    asm.expand_i(
        JoltInstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.expand_i(JoltInstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.expand_i(JoltInstructionKind::LD, v2, v1, 0)?;
    asm.expand_i(JoltInstructionKind::SLLI, v3, v0, 3)?;
    asm.expand_u(JoltInstructionKind::LUI, v0, mask)?;
    asm.expand_r(JoltInstructionKind::SLL, v0, v0, v3)?;
    asm.expand_r(JoltInstructionKind::SLL, v3, rs2(instruction)?, v3)?;
    asm.expand_r(JoltInstructionKind::XOR, v3, v2, v3)?;
    asm.expand_r(JoltInstructionKind::AND, v3, v3, v0)?;
    asm.expand_r(JoltInstructionKind::XOR, v2, v2, v3)?;
    asm.expand_s(JoltInstructionKind::SD, v1, v2, 0)?;
    asm.release_many([v0, v1, v2, v3])?;

    asm.finalize()
}
