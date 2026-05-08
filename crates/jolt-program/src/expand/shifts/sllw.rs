use super::*;

pub(in crate::expand) fn expand_sllw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(
        JoltInstructionKind::VirtualPow2W,
        v_pow2,
        rs2(instruction)?,
        0,
    );
    sequence.emit_r(
        JoltInstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    );
    sequence.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );
    sequence.finish_releasing(allocator, [v_pow2])
}
