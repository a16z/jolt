use super::*;

pub(in crate::expand) fn expand_divu(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;

    asm.expand_j(JoltInstructionKind::VirtualAdvice, v0, 0)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertValidDiv0,
        rs2(instruction)?,
        v0,
        0,
    )?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertMulUNoOverflow,
        v0,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_r(JoltInstructionKind::MUL, v1, v0, rs2(instruction)?)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertLTE,
        v1,
        rs1(instruction)?,
        0,
    )?;
    asm.expand_r(JoltInstructionKind::SUB, v1, rs1(instruction)?, v1)?;
    asm.expand_b(
        JoltInstructionKind::VirtualAssertValidUnsignedRemainder,
        v1,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, v0, 0)?;
    asm.release_many([v0, v1])?;

    asm.finalize()
}
