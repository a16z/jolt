use super::*;

pub(in crate::expand) fn expand_srl(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    );
    sequence.emit_r(
        JoltInstructionKind::VirtualSRL,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    );
    sequence.finish_releasing(allocator, [v_bitmask])
}
