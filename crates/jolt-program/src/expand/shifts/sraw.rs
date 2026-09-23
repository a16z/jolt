use super::*;

/// Lowers variable `SRAW` through a word-width shift mask and fused shift row.
pub(in crate::expand) fn expand_sraw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        Kind::VirtualShiftRightBitmaskW(jolt_riscv::instructions::VirtualShiftRightBitmaskW(())),
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        Kind::VirtualSRAW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_bitmask.operand(),
    );
    asm.release(v_bitmask);

    asm.finalize()
}
