use super::*;

pub(in crate::expand) fn expand_mulw(
    instruction: &SourceRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_r(
        JoltInstructionKind::MUL,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        reg(rs2(instruction)?),
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        reg(rd(instruction)?),
        reg(rd(instruction)?),
        0,
    );

    asm.finalize()
}
