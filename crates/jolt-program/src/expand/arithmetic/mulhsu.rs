use super::*;

/// Lowers signed-by-unsigned `MULHSU` by normalizing the signed operand before
/// reapplying the sign to the high half.
///
/// The sequence computes the absolute value of `rs1`, multiplies it as an
/// unsigned value by `rs2`, and then applies the two's-complement correction
/// needed when `rs1` was negative. The final `SLTU` detects the carry produced
/// while negating the low half, which must be added into the high half.
pub(in crate::expand) fn expand_mulhsu(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v0 = asm.allocate()?;
    let v1 = asm.allocate()?;
    let v2 = asm.allocate()?;
    let v3 = asm.allocate()?;

    asm.emit_i(
        Kind::VirtualMovsign,
        v0.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.emit_i(Kind::ANDI, v1.operand(), v0.operand(), 1);
    asm.emit_r(
        Kind::XOR,
        v2.operand(),
        reg(rs1(instruction)?),
        v0.operand(),
    );
    asm.emit_r(Kind::ADD, v2.operand(), v2.operand(), v1.operand());
    asm.emit_r(
        Kind::MULHU,
        v3.operand(),
        v2.operand(),
        reg(rs2(instruction)?),
    );
    asm.emit_r(
        Kind::MUL,
        v2.operand(),
        v2.operand(),
        reg(rs2(instruction)?),
    );
    asm.emit_r(Kind::XOR, v3.operand(), v3.operand(), v0.operand());
    asm.emit_r(Kind::XOR, v2.operand(), v2.operand(), v0.operand());
    asm.emit_r(Kind::ADD, v0.operand(), v2.operand(), v1.operand());
    asm.emit_r(Kind::SLTU, v0.operand(), v0.operand(), v2.operand());
    asm.emit_r(Kind::ADD, reg(rd(instruction)?), v3.operand(), v0.operand());
    asm.release_many([v0, v1, v2, v3]);

    asm.finalize()
}
