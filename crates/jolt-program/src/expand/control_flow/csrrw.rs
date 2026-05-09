use super::*;

pub(in crate::expand) fn expand_csrrw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = csr_to_virtual_register(csr).ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut asm = ExpansionBuilder::new(*instruction);

    if rd(instruction)? == 0 {
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(virtual_reg),
            reg(rs1(instruction)?),
            0,
        );
        return asm.finalize();
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = asm.allocate()?;
        asm.emit_i(
            JoltInstructionKind::ADDI,
            temp.operand(),
            reg(rs1(instruction)?),
            0,
        );
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(rd(instruction)?),
            reg(virtual_reg),
            0,
        );
        asm.emit_i(
            JoltInstructionKind::ADDI,
            reg(virtual_reg),
            temp.operand(),
            0,
        );
        asm.release(temp)?;
        return asm.finalize();
    }

    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(rd(instruction)?),
        reg(virtual_reg),
        0,
    );
    asm.emit_i(
        JoltInstructionKind::ADDI,
        reg(virtual_reg),
        reg(rs1(instruction)?),
        0,
    );

    asm.finalize()
}
