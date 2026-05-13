use super::*;

pub(in crate::expand) fn expand_srlw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;
    let v_rs1 = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1.operand(),
        reg(rs1(instruction)?),
        1i128 << 32,
    );
    asm.emit_i(
        JoltInstructionKind::ORI,
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        32,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask.operand(),
        v_bitmask.operand(),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRL,
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
    asm.release(v_bitmask);
    asm.release(v_rs1);

    asm.finalize()
}
