use super::*;

pub(in crate::expand) fn expand_sra(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_bitmask = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRA,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    );
    asm.release(v_bitmask)?;

    asm.finalize()
}
