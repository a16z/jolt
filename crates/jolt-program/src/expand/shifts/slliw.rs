use super::*;

/// Lowers `SLLIW` to immediate multiplication followed by RV64 word cleanup.
///
/// The shift amount is restricted to five bits. After the low word has been
/// shifted, `VirtualSignExtendWord` restores the required sign extension from
/// bit 31 into the destination register.
pub(in crate::expand) fn expand_slliw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        1i128 << shift,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );

    asm.finalize()
}
