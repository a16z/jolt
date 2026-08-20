use super::*;

/// Lowers variable `SLLW` through the word-sized power-of-two helper.
///
/// `VirtualPow2W` uses `rs2 & 0x1f`, matching the RV64 word shift rule. The
/// `MULW` truncates and sign-extends the product in the second row.
pub(in crate::expand) fn expand_sllw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_pow2 = asm.allocate()?;

    asm.emit_i(
        Kind::VirtualPow2W,
        v_pow2.operand(),
        reg(rs2(instruction)?),
        0,
    );
    asm.emit_r(
        Kind::MULW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        v_pow2.operand(),
    );
    asm.release(v_pow2);

    asm.finalize()
}
