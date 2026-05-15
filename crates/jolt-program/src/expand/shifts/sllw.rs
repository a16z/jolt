use super::*;

/// Lowers variable `SLLW` through the word-sized power-of-two helper.
///
/// `VirtualPow2W` uses `rs2 & 0x1f`, matching the RV64 word shift rule. The
/// product is then sign-extended from 32 bits so the final row sequence has the
/// same result as the source `SLLW`.
pub(in crate::expand) fn expand_sllw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_pow2 = asm.allocate()?;

    asm.emit_i(
        JoltInstructionKind::VirtualPow2W,
        v_pow2.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        JoltInstructionKind::MUL,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_pow2.operand(),
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord(
            jolt_riscv::instructions::VirtualSignExtendWord(()),
        ),
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );
    asm.release(v_pow2);

    asm.finalize()
}
