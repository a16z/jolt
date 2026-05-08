use super::*;

pub(in crate::expand) fn expand_subw(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_r(
        JoltInstructionKind::SUB,
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
