use super::*;

pub(in crate::expand) fn expand_sllw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_pow2 = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualPow2W,
        v_pow2,
        rs2(instruction)?,
        0,
    );
    asm.emit_r(
        JoltInstructionKind::MUL,
        rd(instruction)?,
        rs1(instruction)?,
        v_pow2,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );
    asm.release(v_pow2)?;

    asm.finalize()
}
