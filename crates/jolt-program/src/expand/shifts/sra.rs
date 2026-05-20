use super::*;

/// Lowers variable arithmetic right shift by first materializing the shift mask.
///
/// `VirtualShiftRightBitmask` encodes `rs2 & 0x3f` as the mask consumed by
/// `VirtualSRA`. Splitting the sequence this way keeps dynamic shift amount
/// checking in the lookup table while preserving the signed right-shift result.
pub(in crate::expand) fn expand_sra(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask(
            jolt_riscv::instructions::VirtualShiftRightBitmask(()),
        ),
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRA,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_bitmask.operand(),
    );
    asm.release(v_bitmask);

    asm.finalize()
}
