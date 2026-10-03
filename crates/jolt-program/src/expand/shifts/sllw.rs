use super::*;

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
