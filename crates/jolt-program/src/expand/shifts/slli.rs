use super::*;

pub(in crate::expand) fn expand_slli(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_op(grammar::ExpansionOp::Row(grammar::RowTemplate::i(
        JoltInstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    )));
    sequence.finish()
}
