use super::*;

/// Lowers variable `SRLW` by embedding the 32-bit logical shift in RV64 space.
///
/// The low word is first moved into the high half. Setting bit 5 of the shift
/// operand makes `VirtualShiftRightBitmask` encode `32 + (rs2 & 0x1f)`, so the
/// logical shift extracts exactly the zero-filled 32-bit result before final
/// word sign extension.
pub(in crate::expand) fn expand_srlw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_bitmask = asm.allocate()?;
    let v_rs1 = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        v_rs1.operand(),
        reg(rs1(instruction)?),
        1i128 << 32,
    );
    asm.emit_i(
        JoltInstructionKind::ORI,
        v_bitmask.operand(),
        reg(rs2(instruction)?),
        32,
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
        JoltInstructionKind::VirtualSRL,
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
    asm.release(v_bitmask);
    asm.release(v_rs1);

    asm.finalize()
}
