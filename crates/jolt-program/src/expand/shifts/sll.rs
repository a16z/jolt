use super::*;

/// Lowers variable `SLL` to a proved power-of-two lookup followed by `MUL`.
///
/// `VirtualPow2` applies the RV64 shift-mask rule to `rs2` and returns
/// `2^(rs2 & 0x3f)`. Multiplying by that value is equivalent to a left shift
/// modulo 2^64, which is the architectural result.
pub(in crate::expand) fn expand_sll(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_pow2 = asm.allocate()?;

    asm.expand_i(
        SourceInstructionKind::VirtualPow2,
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
    asm.release(v_pow2);

    asm.finalize()
}
