use super::*;

pub(in crate::expand) fn expand_sraw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs1 = asm.allocate()?;
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        v_rs1.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_i(
        JoltInstructionKind::ANDI,
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        0x1f,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask.operand(),
        v_bitmask.operand(),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRA,
        reg(rd(instruction)?),
        v_rs1.operand(),
        v_bitmask.operand(),
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );
    asm.release(v_rs1);
    asm.release(v_bitmask);

    asm.finalize()
}
