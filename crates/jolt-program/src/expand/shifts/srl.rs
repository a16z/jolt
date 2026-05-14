use super::*;

/// Lowers variable logical right shift through a dynamic bitmask helper.
///
/// `VirtualShiftRightBitmask` applies the RV64 `rs2 & 0x3f` rule and produces
/// the mask consumed by `VirtualSRL`, which then performs the unsigned shift.
pub(in crate::expand) fn expand_srl(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask,
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRL,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_bitmask.operand(),
    );
    asm.release(v_bitmask);

    asm.finalize()
}
