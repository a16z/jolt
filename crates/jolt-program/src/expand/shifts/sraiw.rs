use super::*;

/// Lowers `SRAIW` to one word-width arithmetic-shift lookup.
pub(in crate::expand) fn expand_sraiw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 32);
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        Kind::VirtualSRAIW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        bitmask as i128,
    );

    asm.finalize()
}
