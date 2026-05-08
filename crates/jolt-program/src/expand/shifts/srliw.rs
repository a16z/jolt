use super::*;

pub(in crate::expand) fn expand_srliw(
    instruction: &NormalizedInstruction,
    allocator: &mut ExpansionAllocator,
) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
    let v_rs1 = allocator.allocate()?;
    let shift = (instruction.operands.imm & 0x1f) + 32;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut sequence = core::ExpansionSequence::new(instruction);
    sequence.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1,
        rs1(instruction)?,
        1i128 << 32,
    );
    sequence.emit_i(
        JoltInstructionKind::VirtualSRLI,
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
