use super::*;

pub(in crate::expand) fn expand_addiw(
    instruction: &NormalizedInstruction,
) -> Result<ExpandedInstructionSequence, ExpansionError> {
    let mut asm = ExpansionBuilder::new(*instruction);

    asm.emit_i(
        JoltInstructionKind::ADDI,
        rd(instruction)?,
        rs1(instruction)?,
        instruction.operands.imm,
    );
    asm.emit_i(
        JoltInstructionKind::VirtualSignExtendWord,
        rd(instruction)?,
        rd(instruction)?,
        0,
    );

    asm.finalize()
}
