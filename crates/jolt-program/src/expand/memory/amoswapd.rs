use super::*;

pub(in crate::expand) fn expand_amoswapd(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;

    asm.expand_i(JoltInstructionKind::LD, v_rd, rs1(instruction)?, 0)?;
    asm.expand_s(
        JoltInstructionKind::SD,
        rs1(instruction)?,
        rs2(instruction)?,
        0,
    )?;
    asm.expand_i(JoltInstructionKind::ADDI, rd(instruction)?, v_rd, 0)?;
    asm.release(v_rd)?;

    asm.finalize()
}
