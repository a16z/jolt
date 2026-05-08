use super::*;

pub(in crate::expand) fn expand_csrrs(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = allocator
        .csr_to_virtual_register(csr)
        .ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    if rs1(instruction)? == 0 {
        asm.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, virtual_reg, 0);
        return asm.finalize();
    } else if rd(instruction)? == 0 {
        asm.emit_r(
            JoltInstructionKind::OR,
            virtual_reg,
            virtual_reg,
            rs1(instruction)?,
        );
        return asm.finalize();
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = asm.allocate()?;
        asm.emit_i(JoltInstructionKind::ADDI, temp, rs1(instruction)?, 0);
        asm.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, virtual_reg, 0);
        asm.emit_r(JoltInstructionKind::OR, virtual_reg, virtual_reg, temp);
        asm.release(temp)?;
        return asm.finalize();
    }

    asm.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, virtual_reg, 0);
    asm.emit_r(
        JoltInstructionKind::OR,
        virtual_reg,
        virtual_reg,
        rs1(instruction)?,
    );

    asm.finalize()
}
