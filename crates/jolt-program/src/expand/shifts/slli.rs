use super::*;

pub(in crate::expand) fn expand_slli(
    instruction: &JoltRow,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x3f;
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        reg(rd(instruction)?),
        reg(rs1(instruction)?),
        1i128 << shift,
    );

    asm.finalize()
}
