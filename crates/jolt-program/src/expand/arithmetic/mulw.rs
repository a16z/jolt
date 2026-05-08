use super::*;

pub(in crate::expand) fn expand_mulw(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_r(
        JoltInstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        rs2(instruction)?,
    );
    sequence.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );
    sequence.finish()
}
