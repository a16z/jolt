use super::*;

pub(in crate::expand) fn expand_mulhsu(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v0 = allocator.allocate()?;
    let v1 = allocator.allocate()?;
    let v2 = allocator.allocate()?;
    let v3 = allocator.allocate()?;
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(
        JoltInstructionKind::VirtualMovsign,
        v0,
        rs1(instruction)?,
        0,
    );
    sequence.emit_i(JoltInstructionKind::ANDI, v1, v0, 1);
    sequence.emit_r(JoltInstructionKind::XOR, v2, rs1(instruction)?, v0);
    sequence.emit_r(JoltInstructionKind::ADD, v2, v2, v1);
    sequence.emit_r(JoltInstructionKind::MULHU, v3, v2, rs2(instruction)?);
    sequence.emit_r(JoltInstructionKind::MUL, v2, v2, rs2(instruction)?);
    sequence.emit_r(JoltInstructionKind::XOR, v3, v3, v0);
    sequence.emit_r(JoltInstructionKind::XOR, v2, v2, v0);
    sequence.emit_r(JoltInstructionKind::ADD, v0, v2, v1);
    sequence.emit_r(JoltInstructionKind::SLTU, v0, v0, v2);
    sequence.emit_r(JoltInstructionKind::ADD, rd(instruction)?, v3, v0);
    sequence.finish_releasing(allocator, [v0, v1, v2, v3])
}
