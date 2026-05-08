use super::*;

pub(in crate::expand) fn expand_sll(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_pow2 = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.expand_i(
        JoltInstructionKind::VirtualPow2,
        v_pow2,
        rs2(instruction)?,
        0,
    )?;
    asm.emit_r(
        JoltInstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    );
    asm.release(v_pow2)?;

    asm.finalize()
}
