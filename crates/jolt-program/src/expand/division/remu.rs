use super::*;

pub(in crate::expand) fn expand_remu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let mut asm = ExpansionBuilder::new(instruction, allocator);

    asm.expand_j(JoltInstructionKind::VirtualAdvice, v0, 0)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_r(JoltInstructionKind::MUL, v0, v0, rs2(instruction)?)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertLTE,
        v0,
        rs1(instruction)?,
        0,
    )?;
    asm.expand_r(JoltInstructionKind::SUB, v0, rs1(instruction)?, v0)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    asm.release(v0)?;

    asm.finalize()
}
