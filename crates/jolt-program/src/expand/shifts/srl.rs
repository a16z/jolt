use super::*;

pub(in crate::expand) fn expand_srl(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask,
        rs2(instruction)?,
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRL,
        rd(instruction)?,
        rs1(instruction)?,
        v_bitmask,
    );
    asm.release(v_bitmask)?;

    asm.finalize()
}
