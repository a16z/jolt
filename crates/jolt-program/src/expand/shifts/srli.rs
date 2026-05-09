use super::*;

pub(in crate::expand) fn expand_srli(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualSRLI,
        rd(instruction)?,
        rs1(instruction)?,
        bitmask as i128,
    );

    asm.finalize()
}
