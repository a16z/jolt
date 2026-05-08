use super::*;

pub(in crate::expand) fn expand_csrrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = allocator
        .csr_to_virtual_register(csr)
        .ok_or(ExpansionError::UnsupportedCsr(csr))?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    if rd(instruction)? == 0 {
        sequence.emit_i(JoltInstructionKind::ADDI, virtual_reg, rs1(instruction)?, 0);
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = allocator.allocate()?;
        sequence.emit_i(JoltInstructionKind::ADDI, temp, rs1(instruction)?, 0);
        sequence.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, virtual_reg, 0);
        sequence.emit_i(JoltInstructionKind::ADDI, virtual_reg, temp, 0);
        return sequence.finish_releasing(allocator, [temp]);
    } else {
        sequence.emit_i(JoltInstructionKind::ADDI, rd(instruction)?, virtual_reg, 0);
        sequence.emit_i(JoltInstructionKind::ADDI, virtual_reg, rs1(instruction)?, 0);
    }
    sequence.finish()
}
