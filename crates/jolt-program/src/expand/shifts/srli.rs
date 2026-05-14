use super::*;

/// Lowers immediate logical right shift to one virtual final row.
///
/// The mask-shaped immediate records the same shift amount as `imm & 0x3f` and
/// lets the final `VirtualSRLI` row perform an unsigned RV64 shift.
pub(in crate::expand) fn expand_srli(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualSRLI,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        bitmask as i128,
    );

    asm.finalize()
}
