use super::*;

pub(in crate::expand) fn expand_mulh(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_sx = asm.allocate()?;
    let v_sy = asm.allocate()?;
    let v_tmp = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v_sx,
        rs1(instruction)?,
        0,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v_sy,
        rs2(instruction)?,
        0,
    );
    asm.emit_r(JoltInstructionKind::MUL, v_sx, v_sx, rs2(instruction)?);
    asm.emit_r(JoltInstructionKind::MUL, v_sy, v_sy, rs1(instruction)?);
    asm.emit_r(
        JoltInstructionKind::MULHU,
        v_tmp,
        rs1(instruction)?,
        rs2(instruction)?,
    );
    asm.emit_r(JoltInstructionKind::ADD, v_tmp, v_tmp, v_sx);
    asm.emit_r(JoltInstructionKind::ADD, rd(instruction)?, v_tmp, v_sy);
    asm.release_many([v_sx, v_sy, v_tmp])?;

    asm.finalize()
}
