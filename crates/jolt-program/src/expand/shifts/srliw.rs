use super::*;

pub(in crate::expand) fn expand_srliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = (instruction.operands.imm & 0x1f) + 32;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    )?;
    asm.emit_i(
        JoltInstructionKind::VirtualSRLI,
        rd(instruction)?,
        v_rs1,
        bitmask as i128,
    )?;
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    )?;
    let sequence = asm.finalize()?;
    allocator.release(v_rs1)?;
    Ok(sequence)
}
