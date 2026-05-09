use super::*;

pub(in crate::expand) fn expand_srlw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;
    let v_rs1 = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    );
    asm.emit_i(JoltInstructionKind::ORI, v_bitmask, rs2(instruction)?, 32);
    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        v_bitmask,
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRL,
        rd(instruction)?,
        v_rs1,
        v_bitmask,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );
    asm.release(v_bitmask)?;
    asm.release(v_rs1)?;

    asm.finalize()
}
