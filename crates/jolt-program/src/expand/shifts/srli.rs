use super::*;

pub(in crate::expand) fn expand_srli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
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
