use super::*;

pub(super) fn expand_slli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    )?;
    asm.finalize()
}

pub(super) fn expand_sll(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualPow2, v_pow2, rs2(instruction)?, 0)?;
    asm.emit_r(
        InstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_pow2)?;
    Ok(sequence)
}

pub(super) fn expand_slliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    asm.finalize()
}

pub(super) fn expand_sllw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(InstructionKind::VirtualPow2W, v_pow2, rs2(instruction)?, 0)?;
    asm.emit_r(
        InstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_pow2)?;
    Ok(sequence)
}

pub(super) fn expand_srl(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRL,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}

pub(super) fn expand_srlw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let v_rs1 = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    )?;
    asm.emit_i(InstructionKind::ORI, v_bitmask, rs2(instruction)?, 32)?;
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRL,
        rd(instruction)?,
        v_rs1,
        v_bitmask,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}

pub(super) fn expand_sra(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRA,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}

pub(super) fn expand_sraw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let v_bitmask = allocator.allocate()?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        v_rs1,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(InstructionKind::ANDI, v_bitmask, rs2(instruction)?, 0x1f)?;
    asm.emit_i(
        InstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    )?;
    asm.emit_r(
        InstructionKind::VirtualSRA,
        rd(instruction)?,
        v_rs1,
        v_bitmask,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    allocator.release(v_bitmask)?;
    Ok(sequence)
}

pub(super) fn expand_srli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSRLI,
        rd(instruction)?,
        rs1(instruction)?,
        bitmask as i128,
    )?;
    asm.finalize()
}

pub(super) fn expand_srliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = (instruction.operands.imm & 0x1f) + 32;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSRLI,
        rd(instruction)?,
        v_rs1,
        bitmask as i128,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}

pub(super) fn expand_srai(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSRAI,
        rd(instruction)?,
        rs1(instruction)?,
        bitmask as i128,
    )?;
    asm.finalize()
}

pub(super) fn expand_sraiw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = instruction.operands.imm & 0x1f;
    let bitmask = right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        v_rs1,
        rs1(instruction)?,
        0,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSRAI,
        rd(instruction)?,
        v_rs1,
        bitmask as i128,
    )?;
    asm.emit_i(
        InstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}

pub(super) fn right_shift_bitmask(shift: u32, len: u32) -> u64 {
    let ones = (1u128 << (len - shift)) - 1;
    (ones << shift) as u64
}
