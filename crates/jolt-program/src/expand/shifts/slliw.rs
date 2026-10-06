use super::*;

pub(in crate::expand) fn expand_slliw(
    instruction: &SourceInstructionRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        Kind::VirtualMULIW,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        1i128 << shift,
    );

    asm.finalize()
}
