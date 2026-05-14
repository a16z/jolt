use super::*;

/// Lowers `AMOSWAP.D` to load-old, store-new, return-old.
///
/// The old doubleword is copied to `rd` after `rs2` has been written to memory,
/// matching the atomic swap's read-modify-write result in Jolt's sequential
/// trace model.
pub(in crate::expand) fn expand_amoswapd(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);
    let v_rd = asm.allocate()?;

    asm.expand_i(
        SourceInstructionKind::LD,
        v_rd.operand(),
        reg(rs1(instruction)?),
        0,
    );
    asm.expand_s(
        SourceInstructionKind::SD,
        reg(rs1(instruction)?),
        reg(rs2(instruction)?),
        0,
    );
    asm.expand_i(
        SourceInstructionKind::ADDI,
        reg(rd(instruction)?),
        v_rd.operand(),
        0,
    );
    asm.release(v_rd);

    asm.finalize()
}
