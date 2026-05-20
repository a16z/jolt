use super::*;

/// Lowers immediate arithmetic right shift to a single virtual final row.
///
/// The immediate is converted into the same bitmask shape used by dynamic
/// shifts. `VirtualSRAI` recovers the shift amount from that mask and performs
/// the signed RV64 shift.
pub(in crate::expand) fn expand_srai(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let bitmask = super::shared::right_shift_bitmask(shift as u32, 64);
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualSRAI,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        bitmask as i128,
    );

    asm.finalize()
}
