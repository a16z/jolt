use super::*;

pub(in crate::expand) fn expand_addiw(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_ops([
        grammar::ExpansionOp::Row(grammar::RowTemplate::i(
            JoltInstructionKind::ADDI,
            rd(instruction)?,
            rs1(instruction)?,
            instruction.operands.imm,
        )),
        grammar::ExpansionOp::Row(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualSignExtendWord,
            rd(instruction)?,
            rd(instruction)?,
            0,
        )),
    ]);
    sequence.finish()
}
