use super::*;

pub(in crate::expand) fn expand_slli(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    )?;
    asm.finalize()
}
