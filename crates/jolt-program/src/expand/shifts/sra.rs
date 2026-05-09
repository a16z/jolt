use super::*;

pub(in crate::expand) fn expand_sra(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRA,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_bitmask.operand(),
    );
    asm.release(v_bitmask);

    asm.finalize()
}
