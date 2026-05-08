use super::*;

pub(in crate::expand) fn expand_csrrw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let csr = csr_address(instruction);
    let virtual_reg = allocator
        .csr_to_virtual_register(csr)
        .ok_or(ExpansionError::UnsupportedCsr(csr))?;
    if rd(instruction)? == 0 {
        return core::ExpansionState::new(allocator).materialize_ops(
            instruction,
            [grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                virtual_reg,
                rs1(instruction)?,
                0,
            ))],
        );
    } else if rd(instruction)? == rs1(instruction)? {
        let temp = allocator.allocate()?;
        return core::ExpansionState::new(allocator).materialize_ops(
            instruction,
            [
                grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                    JoltInstructionKind::ADDI,
                    temp,
                    rs1(instruction)?,
                    0,
                )),
                grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                    JoltInstructionKind::ADDI,
                    rd(instruction)?,
                    virtual_reg,
                    0,
                )),
                grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                    JoltInstructionKind::ADDI,
                    virtual_reg,
                    temp,
                    0,
                )),
                grammar::ExpansionOp::Release(temp),
            ],
        );
    }
    core::ExpansionState::new(allocator).materialize_ops(
        instruction,
        [
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                rd(instruction)?,
                virtual_reg,
                0,
            )),
            grammar::ExpansionOp::Row(grammar::RowTemplate::i(
                JoltInstructionKind::ADDI,
                virtual_reg,
                rs1(instruction)?,
                0,
            )),
        ],
    )
}
