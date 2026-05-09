use super::*;

pub(in crate::expand) fn expand_mulhsu(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v0,
        rs1(instruction)?,
        0,
    );
    asm.emit_i(JoltInstructionKind::ANDI, v1, v0, 1);
    asm.emit_r(JoltInstructionKind::XOR, v2, rs1(instruction)?, v0);
    asm.emit_r(JoltInstructionKind::ADD, v2, v2, v1);
    asm.emit_r(JoltInstructionKind::MULHU, v3, v2, rs2(instruction)?);
    asm.emit_r(JoltInstructionKind::MUL, v2, v2, rs2(instruction)?);
    asm.emit_r(JoltInstructionKind::XOR, v3, v3, v0);
    asm.emit_r(JoltInstructionKind::XOR, v2, v2, v0);
    asm.emit_r(JoltInstructionKind::ADD, v0, v2, v1);
    asm.emit_r(JoltInstructionKind::SLTU, v0, v0, v2);
    asm.emit_r(JoltInstructionKind::ADD, rd(instruction)?, v3, v0);
    asm.release_many([v0, v1, v2, v3])?;

    asm.finalize()
}
