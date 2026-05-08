use super::*;

pub(in crate::expand) fn expand_srai(
    instruction: &NormalizedInstruction,
    _allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(
        JoltInstructionKind::VirtualSRAI,
        rd(instruction)?,
        rs1(instruction)?,
        bitmask as i128,
    );
    sequence.finish()
}
