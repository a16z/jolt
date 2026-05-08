use super::*;

pub(in crate::expand) fn expand_slliw(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_ops([
        grammar::ExpansionOp::Row(grammar::RowTemplate::i(
            JoltInstructionKind::VirtualMULI,
            rd(instruction)?,
            rs1(instruction)?,
            1i128 << shift,
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
