use super::*;

pub(in crate::expand) fn expand_amoswapd(
    instruction: &SourceRow,
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
