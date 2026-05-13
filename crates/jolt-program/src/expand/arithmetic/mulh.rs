use super::*;

pub(in crate::expand) fn expand_mulh(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_sx = asm.allocate()?;
    let v_sy = asm.allocate()?;
    let v_tmp = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v_sx.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v_sy.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::MUL,
        v_sx.operand(),
        v_sx.operand(),
        reg(rs2(instruction)?),
    );
    asm.emit_r(
        JoltInstructionKind::MUL,
        v_sy.operand(),
        v_sy.operand(),
        reg(rs1(instruction)?),
    );
    asm.emit_r(
        JoltInstructionKind::MULHU,
        v_tmp.operand(),
        reg(rs1(instruction)?),
        reg(rs2(instruction)?),
    );
    asm.emit_r(
        JoltInstructionKind::ADD,
        v_tmp.operand(),
        v_tmp.operand(),
        v_sx.operand(),
    );
    asm.emit_r(
        JoltInstructionKind::ADD,
        reg(rd(instruction)?),
        v_tmp.operand(),
        v_sy.operand(),
    );
    asm.release_many([v_sx, v_sy, v_tmp]);

    asm.finalize()
}
