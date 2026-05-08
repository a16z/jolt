use super::*;

pub(in crate::expand) fn expand_csrrs(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = allocator
        .csr_to_virtual_register(csr)
        .ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm =
        assembler::InstrAssembler::new(instruction.address, instruction.is_compressed, allocator);
    if rs1(instruction)? == 0 {
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
    } else if rd(instruction)? == 0 {
        asm.emit_r(
            InstructionKind::OR,
            virtual_reg,
            virtual_reg,
            rs1(instruction)?,
        )?;
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = asm.allocator().allocate()?;
        asm.emit_i(InstructionKind::ADDI, temp, rs1(instruction)?, 0)?;
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
        asm.emit_r(InstructionKind::OR, virtual_reg, virtual_reg, temp)?;
        asm.allocator().release(temp)?;
    } else {
        asm.emit_i(InstructionKind::ADDI, rd(instruction)?, virtual_reg, 0)?;
        asm.emit_r(
            InstructionKind::OR,
            virtual_reg,
            virtual_reg,
            rs1(instruction)?,
        )?;
    }
    asm.finalize()
}
