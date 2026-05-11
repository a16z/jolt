use common::constants::RAM_START_ADDRESS;

use super::*;

pub(in crate::expand) fn emit_ram_region_assertion(
    asm: &mut assembler::InstrAssembler<'_>,
    address_register: u8,
) -> Result<(), ExpansionError> {
    let ram_start = asm.allocator().allocate()?;
    asm.emit_u(InstructionKind::LUI, ram_start, RAM_START_ADDRESS as i128)?;
    asm.emit_b(
        InstructionKind::VirtualAssertLTE,
        ram_start,
        address_register,
        0,
    )?;
    asm.allocator().release(ram_start)?;
    Ok(())
}

pub(in crate::expand) fn expand_byte_load(
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

pub(in crate::expand) fn expand_halfword_load(
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

pub(in crate::expand) fn expand_advice_load(
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

pub(in crate::expand) fn expand_amo_d(
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

pub(in crate::expand) fn expand_amo_minmax_d(
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

pub(in crate::expand) fn expand_amo_w(
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

pub(in crate::expand) fn expand_amo_minmax_w(
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

pub(in crate::expand) fn amo_pre64(
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
pub(in crate::expand) fn amo_post64(
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

pub(in crate::expand) fn expand_narrow_store(
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
