use super::*;

pub(in crate::expand) fn expand_slliw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let shift = instruction.operands.imm & 0x1f;
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::VirtualMULI,
        rd(instruction)?,
        rs1(instruction)?,
        1i128 << shift,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );

    asm.finalize()
}
