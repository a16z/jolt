use super::*;

/// Lowers variable `SRAW` as a signed 32-bit shift embedded in RV64 rows.
///
/// The low word of `rs1` is sign-extended before shifting, `rs2` is masked to
/// five bits, and the output is sign-extended again. This prevents unrelated
/// high bits of `rs1` or `rs2` from influencing the architectural word result.
pub(in crate::expand) fn expand_sraw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rs1 = asm.allocate()?;
    let v_bitmask = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
        v_rs1.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_i(
        JoltInstructionKind::ANDI,
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        0x1f,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualShiftRightBitmask(
            jolt_riscv::instructions::VirtualShiftRightBitmask(()),
        ),
        v_bitmask.operand(),
        v_bitmask.operand(),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::VirtualSRA,
        reg(rd(instruction)?),
        v_rs1.operand(),
        v_bitmask.operand(),
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );
    asm.release(v_rs1);
    asm.release(v_bitmask);

    asm.finalize()
}
