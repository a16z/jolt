use super::*;

pub(in crate::expand) fn expand_sraiw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = instruction.operands.imm & 0x1f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        v_rs1,
        rs1(instruction)?,
        0,
    );
    sequence.emit_i(
        JoltInstructionKind::VirtualSRAI,
        rd(instruction)?,
        v_rs1,
        bitmask as i128,
    );
    sequence.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );
    sequence.finish_releasing(allocator, [v_rs1])
}
