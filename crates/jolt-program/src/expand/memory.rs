use super::*;

pub(super) fn expand_lb(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_byte_load(instruction, allocator, true)
}

pub(super) fn expand_lbu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_byte_load(instruction, allocator, false)
}

pub(super) fn expand_byte_load(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    signed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 7)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(
        if signed {
            InstructionKind::SRAI
        } else {
            InstructionKind::SRLI
        },
        rd(instruction)?,
        v1,
        56,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

pub(super) fn expand_lh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_halfword_load(instruction, allocator, true)
}

pub(super) fn expand_lhu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_halfword_load(instruction, allocator, false)
}

pub(super) fn expand_halfword_load(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    signed: bool,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertHalfwordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 6)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(
        if signed {
            InstructionKind::SRAI
        } else {
            InstructionKind::SRLI
        },
        rd(instruction)?,
        v1,
        48,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

pub(super) fn expand_lw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SRL, v1, v1, v0)?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        v1,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

pub(super) fn expand_lwu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v1, v1, 0)?;
    asm.emit_i(InstructionKind::XORI, v0, v0, 4)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_r(InstructionKind::SLL, v1, v1, v0)?;
    asm.emit_i(InstructionKind::SRLI, rd(instruction)?, v1, 32)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    Ok(sequence)
}

pub(super) fn expand_lrd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_d = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::LD, rd(instruction)?, rs1(instruction)?, 0)?;
    asm.finalize()
}

pub(super) fn expand_lrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation_w = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, rs1(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, 0, 0)?;
    asm.emit_i(InstructionKind::LW, rd(instruction)?, rs1(instruction)?, 0)?;
    asm.finalize()
}

pub(super) fn expand_advice_load(
    instruction: &NormalizedInstruction,
    byte_len: i128,
    sign_extension_shift: Option<i128>,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_j(
        InstructionKind::VirtualAdviceLoad,
        rd(instruction)?,
        byte_len,
    )?;
    if let Some(shift) = sign_extension_shift {
        asm.emit_i(
            InstructionKind::SLLI,
            rd(instruction)?,
            rd(instruction)?,
            shift,
        )?;
        asm.emit_i(
            InstructionKind::SRAI,
            rd(instruction)?,
            rd(instruction)?,
            shift,
        )?;
    }
    asm.finalize()
}

pub(super) fn expand_amo_d(
    instruction: &NormalizedInstruction,
    op: InstructionKind,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs2 = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.emit_r(op, v_rs2, v_rd, rs2(instruction)?)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v_rs2, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs2)?;
    allocator.release(v_rd)?;
    Ok(sequence)
}

pub(super) fn expand_amoswapd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, rs2(instruction)?, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    Ok(sequence)
}

pub(super) fn expand_amo_minmax_d(
    instruction: &NormalizedInstruction,
    compare_op: InstructionKind,
    min: bool,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::LD, v0, rs1(instruction)?, 0)?;
    let (cmp_rs1, cmp_rs2) = if min {
        (rs2(instruction)?, v0)
    } else {
        (v0, rs2(instruction)?)
    };
    asm.emit_r(compare_op, v1, cmp_rs1, cmp_rs2)?;
    asm.emit_r(InstructionKind::SUB, v2, rs2(instruction)?, v0)?;
    asm.emit_r(InstructionKind::MUL, v2, v2, v1)?;
    asm.emit_r(InstructionKind::ADD, v1, v0, v2)?;
    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v1, 0)?;
    asm.emit_i(InstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    Ok(sequence)
}

pub(super) fn expand_amo_w(
    instruction: &NormalizedInstruction,
    op: InstructionKind,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let v_rs2 = allocator.allocate()?;
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    asm.emit_r(op, v_rs2, v_rd, rs2(instruction)?)?;
    amo_post64(
        &mut asm,
        rs1(instruction)?,
        v_rs2,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    allocator.release(v_rs2)?;
    allocator.release(v_mask)?;
    allocator.release(v_dword)?;
    allocator.release(v_shift)?;
    Ok(sequence)
}

pub(super) fn expand_amoswapw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_mask = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let v_rd = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;
    amo_post64(
        &mut asm,
        rs1(instruction)?,
        rs2(instruction)?,
        v_dword,
        v_shift,
        v_mask,
        rd(instruction)?,
        v_rd,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_mask)?;
    allocator.release(v_dword)?;
    allocator.release(v_shift)?;
    allocator.release(v_rd)?;
    Ok(sequence)
}

pub(super) fn expand_amo_minmax_w(
    instruction: &NormalizedInstruction,
    compare_op: InstructionKind,
    min: bool,
    signed: bool,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rd = allocator.allocate()?;
    let v_dword = allocator.allocate()?;
    let v_shift = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    amo_pre64(&mut asm, rs1(instruction)?, v_rd, v_dword, v_shift)?;

    let v_rs2 = asm.allocator().allocate()?;
    let v0 = asm.allocator().allocate()?;
    let extend_op = if signed {
        InstructionKind::VirtualSignExtendWord
    } else {
        InstructionKind::VirtualZeroExtendWord
    };
    asm.emit_i(extend_op, v_rs2, rs2(instruction)?, 0)?;
    asm.emit_i(extend_op, v0, v_rd, 0)?;
    let (cmp_rs1, cmp_rs2) = if min { (v_rs2, v0) } else { (v0, v_rs2) };
    asm.emit_r(compare_op, v0, cmp_rs1, cmp_rs2)?;
    asm.emit_r(InstructionKind::SUB, v_rs2, rs2(instruction)?, v_rd)?;
    asm.emit_r(InstructionKind::MUL, v_rs2, v_rs2, v0)?;
    asm.emit_r(InstructionKind::ADD, v_rs2, v_rs2, v_rd)?;
    amo_post64(
        &mut asm,
        rs1(instruction)?,
        v_rs2,
        v_dword,
        v_shift,
        v0,
        rd(instruction)?,
        v_rd,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rd)?;
    allocator.release(v_dword)?;
    allocator.release(v_shift)?;
    allocator.release(v_rs2)?;
    allocator.release(v0)?;
    Ok(sequence)
}

pub(super) fn amo_pre64(
    asm: &mut assembler::InstrAssembler<'_>,
    rs1: u8,
    v_rd: u8,
    v_dword: u8,
    v_shift: u8,
) -> Result<(), ExpansionError> {
    asm.emit_align(InstructionKind::VirtualAssertWordAlignment, rs1, 0)?;
    asm.emit_i(InstructionKind::ANDI, v_shift, rs1, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v_dword, v_shift, 0)?;
    asm.emit_i(InstructionKind::SLLI, v_shift, rs1, 3)?;
    asm.emit_r(InstructionKind::SRL, v_rd, v_dword, v_shift)?;
    Ok(())
}

#[expect(clippy::too_many_arguments)]
pub(super) fn amo_post64(
    asm: &mut assembler::InstrAssembler<'_>,
    rs1: u8,
    v_rs2: u8,
    v_dword: u8,
    v_shift: u8,
    v_mask: u8,
    rd: u8,
    v_rd: u8,
) -> Result<(), ExpansionError> {
    asm.emit_i(InstructionKind::ORI, v_mask, 0, format_i_imm(-1))?;
    asm.emit_i(InstructionKind::SRLI, v_mask, v_mask, 32)?;
    asm.emit_r(InstructionKind::SLL, v_mask, v_mask, v_shift)?;
    asm.emit_r(InstructionKind::SLL, v_shift, v_rs2, v_shift)?;
    asm.emit_r(InstructionKind::XOR, v_shift, v_dword, v_shift)?;
    asm.emit_r(InstructionKind::AND, v_shift, v_shift, v_mask)?;
    asm.emit_r(InstructionKind::XOR, v_dword, v_dword, v_shift)?;
    asm.emit_i(InstructionKind::ANDI, v_mask, rs1, format_i_imm(-8))?;
    asm.emit_s(InstructionKind::SD, v_mask, v_dword, 0)?;
    asm.emit_i(InstructionKind::VirtualSignExtendWord, rd, v_rd, 0)?;
    Ok(())
}

pub(super) fn expand_sb(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_narrow_store(instruction, allocator, 0xff, None)
}

pub(super) fn expand_sh(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    expand_narrow_store(
        instruction,
        allocator,
        0xffff,
        Some(InstructionKind::VirtualAssertHalfwordAlignment),
    )
}

pub(super) fn expand_narrow_store(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
    mask: i128,
    alignment: Option<InstructionKind>,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    if let Some(alignment) = alignment {
        asm.emit_align(alignment, rs1(instruction)?, instruction.operands.imm)?;
    }
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v2, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v3, v0, 3)?;
    asm.emit_u(InstructionKind::LUI, v0, mask)?;
    asm.emit_r(InstructionKind::SLL, v0, v0, v3)?;
    asm.emit_r(InstructionKind::SLL, v3, rs2(instruction)?, v3)?;
    asm.emit_r(InstructionKind::XOR, v3, v2, v3)?;
    asm.emit_r(InstructionKind::AND, v3, v3, v0)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v3)?;
    asm.emit_s(InstructionKind::SD, v1, v2, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}

pub(super) fn expand_sw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_align(
        InstructionKind::VirtualAssertWordAlignment,
        rs1(instruction)?,
        instruction.operands.imm,
    )?;
    asm.emit_i(
        InstructionKind::ADDI,
        v0,
        rs1(instruction)?,
        format_i_imm(instruction.operands.imm),
    )?;
    asm.emit_i(InstructionKind::ANDI, v1, v0, format_i_imm(-8))?;
    asm.emit_i(InstructionKind::LD, v2, v1, 0)?;
    asm.emit_i(InstructionKind::SLLI, v0, v0, 3)?;
    asm.emit_i(InstructionKind::ORI, v3, 0, format_i_imm(-1))?;
    asm.emit_i(InstructionKind::SRLI, v3, v3, 32)?;
    asm.emit_r(InstructionKind::SLL, v3, v3, v0)?;
    asm.emit_r(InstructionKind::SLL, v0, rs2(instruction)?, v0)?;
    asm.emit_r(InstructionKind::XOR, v0, v2, v0)?;
    asm.emit_r(InstructionKind::AND, v0, v0, v3)?;
    asm.emit_r(InstructionKind::XOR, v2, v2, v0)?;
    asm.emit_s(InstructionKind::SD, v1, v2, 0)?;
    let sequence = asm.finalize()?;
    allocator.release(v0)?;
    allocator.release(v1)?;
    allocator.release(v2)?;
    allocator.release(v3)?;
    Ok(sequence)
}

pub(super) fn expand_scd(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_d_register();
    let v_reservation_w = allocator.reservation_w_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let v_success = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::VirtualAdvice, v_success, 0)?;

    let v_one = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::ADDI, v_one, 0, 1)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v_success, v_one, 0)?;
    asm.allocator().release(v_one)?;

    let v_addr_diff = asm.allocator().allocate()?;
    asm.emit_r(
        InstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
    )?;
    asm.emit_r(InstructionKind::MUL, v_addr_diff, v_success, v_addr_diff)?;
    asm.emit_b(InstructionKind::VirtualAssertEQ, v_addr_diff, 0, 0)?;
    asm.allocator().release(v_addr_diff)?;

    let v_mem = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::LD, v_mem, rs1(instruction)?, 0)?;

    let v_diff = asm.allocator().allocate()?;
    asm.emit_r(InstructionKind::SUB, v_diff, rs2(instruction)?, v_mem)?;
    asm.emit_r(InstructionKind::MUL, v_diff, v_diff, v_success)?;
    asm.emit_r(InstructionKind::ADD, v_diff, v_mem, v_diff)?;
    asm.allocator().release(v_mem)?;

    asm.emit_s(InstructionKind::SD, rs1(instruction)?, v_diff, 0)?;
    asm.allocator().release(v_diff)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation, 0, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_w, 0, 0)?;
    asm.emit_i(InstructionKind::XORI, rd(instruction)?, v_success, 1)?;
    asm.allocator().release(v_success)?;

    asm.finalize()
}

pub(super) fn expand_scw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_reservation = allocator.reservation_w_register();
    let v_reservation_d = allocator.reservation_d_register();
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);

    let v_success = asm.allocator().allocate()?;
    asm.emit_j(InstructionKind::VirtualAdvice, v_success, 0)?;

    let v_one = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::ADDI, v_one, 0, 1)?;
    asm.emit_b(InstructionKind::VirtualAssertLTE, v_success, v_one, 0)?;
    asm.allocator().release(v_one)?;

    let v_addr_diff = asm.allocator().allocate()?;
    asm.emit_r(
        InstructionKind::SUB,
        v_addr_diff,
        v_reservation,
        rs1(instruction)?,
    )?;
    asm.emit_r(InstructionKind::MUL, v_addr_diff, v_success, v_addr_diff)?;
    asm.emit_b(InstructionKind::VirtualAssertEQ, v_addr_diff, 0, 0)?;
    asm.allocator().release(v_addr_diff)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation, v_success, 0)?;
    asm.allocator().release(v_success)?;

    let v_mem = asm.allocator().allocate()?;
    asm.emit_i(InstructionKind::LW, v_mem, rs1(instruction)?, 0)?;

    let v_diff = asm.allocator().allocate()?;
    asm.emit_r(InstructionKind::SUB, v_diff, rs2(instruction)?, v_mem)?;
    asm.emit_r(InstructionKind::MUL, v_diff, v_diff, v_reservation)?;
    asm.emit_r(InstructionKind::ADD, v_diff, v_mem, v_diff)?;
    asm.allocator().release(v_mem)?;

    asm.emit_i(InstructionKind::ADDI, v_reservation_d, v_diff, 0)?;
    asm.allocator().release(v_diff)?;

    asm.emit_s(InstructionKind::SW, rs1(instruction)?, v_reservation_d, 0)?;

    asm.emit_i(InstructionKind::XORI, rd(instruction)?, v_reservation, 1)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation, 0, 0)?;
    asm.emit_i(InstructionKind::ADDI, v_reservation_d, 0, 0)?;

    asm.finalize()
}
