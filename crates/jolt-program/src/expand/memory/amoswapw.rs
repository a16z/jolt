use jolt_riscv::SourceInstructionKind as Kind;

use super::shared::{AccessWidth, LaneWindow};
use super::*;

/// Lowers `AMOSWAP.W` as a read-modify-write of the word lane: the old word is
/// extracted sign-extended, the low word of `rs2` is merged back and stored,
/// and `rd` receives the old word last because it may alias `rs1` or `rs2`.
pub(in crate::expand) fn expand_amoswapw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let doubleword = asm.allocate()?;
    let lane = asm.allocate()?;
    let old = asm.allocate()?;

    let window = LaneWindow::open(
        &mut asm,
        AccessWidth::Word,
        doubleword.operand(),
        lane.operand(),
        reg(rs1(instruction)?),
        0,
    );
    window.extract(&mut asm, old.operand(), true);
    window.merge(&mut asm, reg(rs2(instruction)?));
    asm.emit_i(Kind::ADDI, reg(rd(instruction)?), old.operand(), 0);
    asm.release_many([doubleword, lane, old]);

    asm.finalize()
}
